import sys
sys.dont_write_bytecode = True  # Prevent __pycache__ directory being created.

from collections import defaultdict

from tensordict import TensorDict, TensorDictBase
import torch
import torch.multiprocessing as mp
import py_interface
from ctypes import Structure, c_double, c_uint64, c_uint32, sizeof
import multiprocessing
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torch.func import stack_module_state
import copy
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import snntorch as snn
from snntorch import surrogate
from enum import Enum
import torch.nn.functional as F
from torchrl.data import Binary, Bounded, LazyMemmapStorage, SliceSampler, TensorDictReplayBuffer, Unbounded
from torchrl.envs import EnvBase, ExplorationType, set_exploration_type
from torchrl.data import Composite
from torchrl.collectors import SyncDataCollector
from torch.distributions import Categorical, Distribution
from tqdm import tqdm
import subprocess
import os
import json
import signal
import psutil
import time
import gc
from snntorch import spikegen
import argparse
import matplotlib.pyplot as plt
import math

from delta_modulator import DeltaModulator
from rate_encoder import RateEncoder

class OBS(Enum):
    availableBW = 0
    averageQLength = 2
    txRateECN = 3
    k_min_in = 4
    k_delta_in = 5
    p_max_in = 6

class ENCODING(Enum):
    DIRECT = 0
    DELTA = 1
    RATE = 2

parser = argparse.ArgumentParser(description="RL parser")

parser.add_argument("--network_type", type=str, default="ann", help="ANN or SNN")
parser.add_argument("--eval", action="store_true", help="Eval mode")
parser.add_argument("--batch_offset", type=int, default=0, help="Batch offset for eval")
parser.add_argument("--hidden_layers", type=int, default=1, help="Num hidden layers")
parser.add_argument("--hidden_neurons", type=int, default=64, help="Num hidden cells")
parser.add_argument("--enc_in", type=str, default="direct", help="Encoding in - only for snn")
parser.add_argument("--enc_out", type=str, default="direct", help="Encoding out - only for snn")
parser.add_argument("--beta", type=float, default=0.8, help="Beta used in LIF neurons")
parser.add_argument("--firing_rate", type=float, default=0.4, help="Firing rate to be used")
parser.add_argument("--cont_or_disc", type=str, default="continuous", help="Continuous or discrete action space")
parser.add_argument("--mse_loss_scale", type=float, default=20, help="Scaling for MSE Loss for direct encoding")
parser.add_argument("--sim_num", type=int, default=0, help="Sim number to run simultaneous trainings")
parser.add_argument("--topo", type=str, default="star", help="Topology to use: star or fat")
args = parser.parse_args()

BATCH_OFFSET = args.batch_offset
NUM_HIDDEN_LAYERS = args.hidden_layers
NUM_HIDDEN_CELLS = args.hidden_neurons
INPUT_ENCODING = ENCODING[args.enc_in.upper()]
OUTPUT_ENCODING = ENCODING[args.enc_out.upper()]
PERFORM_EVAL = args.eval
SNN_BETA = args.beta
FIRING_RATE = args.firing_rate
CONT_OR_DISC = args.cont_or_disc.upper()
MSE_LOSS_SCALING = args.mse_loss_scale
SIM_NUM = args.sim_num
TOPO = args.topo.upper()

if TOPO == "STAR" or TOPO == "TESTING":
    NUM_AGENTS = 1
elif TOPO == "FAT":
    NUM_AGENTS = 56  # This topology has 56 switches in a fat tree topology (spine-leaf topology)
else:
    raise Exception("topo must be either 'star', 'fat', or 'testing'.")


print(f"SIM NUM = {SIM_NUM}", flush=True)

if CONT_OR_DISC not in ["CONTINUOUS", "DISCRETE"]:
    raise Exception("cont_or_disc must be either 'continuous' or 'discrete'.")

NUM_PORTS = 256
MOMENTUM = 0.01

if args.network_type.lower() == "snn":
    USE_SNN = True
else:
    USE_SNN = False

USE_MAPPO = True

if USE_MAPPO:
    NUM_NETWORKS = 1  # Number of independent value networks - using same networks for all agents for value networks for MAPPO (as suggested in 'SurprisingEffectivenessOfPPO').
else:
    NUM_NETWORKS = NUM_AGENTS

print("============ CURRENT CONFIG ==============")
print(f"NETWORK: {'snn' if USE_SNN else 'ann'}", flush=True)

if USE_SNN:
    print(f"INPUT_ENCODING: {INPUT_ENCODING}", flush=True)
    print(f"OUTPUT_ENCODING: {OUTPUT_ENCODING}", flush=True)

print(f"PERFORM_EVAL: {PERFORM_EVAL}", flush=True)
print(f"BATCH_OFFSET: {BATCH_OFFSET}", flush=True)
print(f"NUM_HIDDEN_LAYERS: {NUM_HIDDEN_LAYERS}", flush=True)
print(f"NUM_HIDDEN_CELLS: {NUM_HIDDEN_CELLS}", flush=True)
print("==========================================")

NUM_ACTIONS = 3
NUM_STATE_PARAMS = 6

# discrete action space for ECN parameters
ACTION_BINS = [10, 10, 21]

if CONT_OR_DISC == "CONTINUOUS":
    OUT_FEATURES = 2 * NUM_ACTIONS
else:
    OUT_FEATURES = sum(ACTION_BINS)

FIRST_LAYERS_SPIKING_AVERAGE = 0
FIRST_LAYERS_SPIKING_NUM_SAMPLES = 0

TOTAL_FRAMES = 12_000_000
SEQUENCE_LENGTH = 16

# N.B. NUM_DESIRED_SEQUENCES must be divisible by NUM_MINI_BATCHES.
NUM_DESIRED_SEQUENCES = 8
NUM_MINI_BATCHES = 2

if NUM_DESIRED_SEQUENCES % NUM_MINI_BATCHES != 0:
    raise Exception("NUM_DESIRED_SEQUENCES must be divisible by NUM_MINI_BATCHES.")

FIRST_LAYERS_SPIKING_AVERAGE = 0 
FIRST_LAYERS_SPIKING_NUM_SAMPLES = 0

class SplitMultiCategorical(Distribution):
    def __init__(self, logits):
        # Split the flat tensor into three differently sized tensors
        self.logits_split = torch.split(logits, ACTION_BINS, dim=-1)
        
        # Create 3 independent Categorical distributions
        self.dists = [Categorical(logits=l) for l in self.logits_split]

        batch_shape = logits.shape[:-1] 
        event_shape = torch.Size([len(ACTION_BINS)])
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        # value shape: [..., 3]
        log_probs = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(value[..., i]))
            
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self):
        return sum([dist.entropy() for dist in self.dists])
    
    @property
    def mode(self):
        return torch.stack([dist.mode for dist in self.dists], dim=-1)
    
    @property
    def mean(self):
        return self.mode
        
    def sample(self, sample_shape=torch.Size()):
        return torch.stack([dist.sample(sample_shape) for dist in self.dists], dim=-1)

class SNNNetwork(nn.Module):
    def __init__(self, num_hidden_layers, num_hidden, out_features, encoding_in, encoding_out, beta, saved_models_suffix, virtual_timesteps, device, debug_eval = False):
        super().__init__()
        self.device = device
        if encoding_out != ENCODING.DIRECT:
            raise Exception("Only Direct Encoding is currently supported for the output.")

        self.encoding_in = encoding_in  # Enum
        
        init_thresholds = torch.zeros(NUM_STATE_PARAMS, device=device)
        adaptive_threshold_mask = torch.tensor([1, 1, 1, 1, 1, 1], device=device)
        if encoding_in == ENCODING.DELTA:
            init_thresholds = torch.tensor([0.5] * 6, device=device)
            
        init_scaling_factor = 1
        if encoding_in == ENCODING.RATE:
            init_scaling_factor = torch.tensor([0.5] * 6, device=device)

        window_size = 5000
        if encoding_in == ENCODING.RATE:
            self.virtual_timesteps = virtual_timesteps
        else:
            self.virtual_timesteps = 1

        self.delta_modulator = DeltaModulator(
            init_thresholds=init_thresholds,
            val_names=["availableBW","averageQLength","txRateECN","k_min","k_delta","p_max"],
            learning_rate=0.2,
            window_size=window_size,  # Approximately update thresholds once every 5 batches.
            saved_models_suffix=saved_models_suffix,
            device=device,
            spike_density=FIRING_RATE,
            adaptive_threshold_mask=adaptive_threshold_mask
        )

        self.rate_encoder = RateEncoder(
            window_size=window_size,
            virtual_timesteps=self.virtual_timesteps,
            learning_rate=0.2,
            spike_rate=FIRING_RATE,
            device=device,
            scaling_factor=init_scaling_factor,
            adaptive_sf_mask=torch.ones(NUM_STATE_PARAMS, device=device)
        )

        # initialize layers
        self.num_hidden = num_hidden
        BETA = beta  # N.B. to make this a learnable parameter, wrap in nn.Parameter
        self.fc1 = nn.Linear(NUM_STATE_PARAMS, num_hidden)

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden

        self.hidden_and_output_layers = []
        for i in range(num_hidden_layers):
            self.hidden_and_output_layers.append(snn.Leaky(beta=BETA))

            if i != num_hidden_layers - 1:
                self.hidden_and_output_layers.append(nn.Linear(num_hidden, num_hidden))
            else:
                self.hidden_and_output_layers.append(nn.Linear(num_hidden, out_features))
        
        self.hidden_and_output_layers.append(snn.Leaky(beta=BETA, reset_mechanism="none"))
        self.hidden_and_output_layers = nn.ModuleList(self.hidden_and_output_layers)

        self.debug_eval = debug_eval
        self.mem_potentials = []
        self.out_features = out_features
        self.step_count = 0

        self.rolling_mean = None
        self.rolling_variance = None
        self.momentum = MOMENTUM

        self.firing_rate_hist = []

    def reset_firing_rates(self):
        global FIRST_LAYERS_SPIKING_AVERAGE, FIRST_LAYERS_SPIKING_NUM_SAMPLES
        FIRST_LAYERS_SPIKING_AVERAGE = 0
        FIRST_LAYERS_SPIKING_NUM_SAMPLES = 0

    def track_firing_rates(self, spk):
        global FIRST_LAYERS_SPIKING_AVERAGE, FIRST_LAYERS_SPIKING_NUM_SAMPLES
        if self.encoding_in == ENCODING.DIRECT:
            sum_spks = spk.sum().item()

            FIRST_LAYERS_SPIKING_AVERAGE = ((spk.mean().item() + FIRST_LAYERS_SPIKING_NUM_SAMPLES * FIRST_LAYERS_SPIKING_AVERAGE)) / (FIRST_LAYERS_SPIKING_NUM_SAMPLES + 1)
            FIRST_LAYERS_SPIKING_NUM_SAMPLES += 1

    def normalise(self, x, update, mask, mean=None, std=None):
        if update:
            if self.rolling_mean is None and self.rolling_variance is None:
                self.rolling_mean = x.clone()
                self.rolling_variance = torch.ones_like(x)

                return torch.zeros_like(x), self.rolling_mean, self.rolling_variance
            else:
                prev_mean = self.rolling_mean.clone()
                self.rolling_mean = (1 - self.momentum) * self.rolling_mean + self.momentum * x
                self.rolling_variance = (1 - self.momentum) * self.rolling_variance + self.momentum * (x - prev_mean) * (x - self.rolling_mean)
                std_dev = torch.sqrt(self.rolling_variance + 1e-8)  # Prevent division by zero in next line.
                x_norm = (x - self.rolling_mean) / std_dev

                return x_norm, self.rolling_mean, std_dev
        else:
            if mean is None or std is None or mask is None:
                raise Exception("Mean and std dev of rollout must be provided in forward_sequences")
            
            x_norm = (x - mean) / std
            x_norm = x_norm * mask
            return x_norm
        

    # TODO: rename delta_thresholds to something more suitable as it is used for both delta encoding and rate encoding.
    def forward(self, x, mask, hidden_states, prev_obs, readout_states, delta_thresholds, means, stds):
        if hidden_states is None:
            raise Exception("hidden_states shouldn't be None when using pytorch environment as a zeroed array is returned in _reset. If using SNNNetwork without rl environment, proceed.")
        
        mem = hidden_states

        if isinstance(mem, torch.Tensor):
            mem = mem.to(x.device)

        
        if x.dim() < 4:  # x: [Port, Features]
            # used for data/transition collection
            cur, mask, h, new_prev_obs, new_readout_states, current_thresholds, means, stds =  self.forward_step(x, mask, mem, prev_obs, readout_states)
            return cur, mask, h, new_prev_obs, new_readout_states, current_thresholds, means, stds

        else:  # x: [Sequence, Time, Port, Features]
            # used when training
            cur, mask, h, new_prev_obs, new_readout_states, current_thresholds = self.forward_sequences(x, mask, mem, prev_obs, readout_states, delta_thresholds, means, stds)
            return cur, mask, h, new_prev_obs, new_readout_states, current_thresholds, means, stds

    def forward_step(self, x, mask, mem, prev_obs, readout_states):
        """
        Used in the data collection phase where the collector processes individual transitions.
        """
        new_prev_obs = x.clone()
        current_thresholds = None

        self.step_count += 1

        mem = mem.clone()
        readout_states = readout_states.clone()

        if self.encoding_in == ENCODING.DELTA:
            x, means, stds = self.normalise(x, update=True, mask=mask)
            x = self.delta_modulator.encode(cur_vals=x, mask=mask)
            current_thresholds = self.delta_modulator.thresholds.clone()
        elif self.encoding_in == ENCODING.RATE:
            x, means, stds = self.normalise(x, update=True, mask=mask)
            x = torch.sigmoid(x) # Ensure the rate is positive by passing it through sigmoid.
            x = self.rate_encoder.encode(x, mask)
            current_thresholds = torch.zeros_like(x)
            current_thresholds = self.rate_encoder.scaling_factors.clone()
        elif self.encoding_in == ENCODING.DIRECT:
            means = torch.zeros_like(x)
            stds = torch.zeros_like(x)


        for vt in range(self.virtual_timesteps):
            if INPUT_ENCODING == ENCODING.RATE:
                in_data = x[vt, ...]
            else:
                in_data = x

            cur = self.fc1(in_data)
            encoding_layer_spikes = []  # N.b. this is only used for direct encoding, which has 1 virtual timestep.
            spk = None
            # Loop through the layers up to the output layer.
            for i in range(len(self.hidden_and_output_layers) - 1):
                if i % 2 == 0:  # Spiking layer
                    mem_idx = i // 2  # As linear layers don't have a membrane potential - must index these separately.
                    current_mem = mem[..., mem_idx, :]
                    spk, new_mem = self.hidden_and_output_layers[i](cur, current_mem)
                    
                    if i == 0:
                        self.track_firing_rates(spk)
                        encoding_layer_spikes.append(spk)

                    if self.debug_eval:
                        print(f"Layer {mem_idx}, spikes: {spk.sum()}", flush=True)
                    mem[..., mem_idx, :] = new_mem
                else:  # Linear layer
                    cur = self.hidden_and_output_layers[i](spk)

        self.current_encoding_rate = torch.stack(encoding_layer_spikes).mean()
        self.firing_rate_hist.append(self.current_encoding_rate)

        hidden_states = mem 
        # readout_states is the membrane potential of the output layer.
        _, new_readout_states = self.hidden_and_output_layers[-1](cur, readout_states)

        if self.debug_eval:
            print(f"Readout: {new_readout_states}", flush=True)

        if current_thresholds is None:
            current_thresholds = torch.zeros_like(x)

        return new_readout_states, mask, hidden_states, new_prev_obs, new_readout_states, current_thresholds, means, stds

    def forward_sequences(self, x, mask, hidden_states, prev_obs, readout_states, current_thresholds, means, stds):
        """
        Used by the loss module for a batch of sequences.
        """
        continuous_x = x
        mem = hidden_states[:, 0, ...].clone()  # Only get the first hidden state as the others will be outdated.
        readout_states = readout_states[:, 0, ...].clone()  # Only take the first readout_state as the subsequent ones in sequence will be determined from this initial one.

        batch_size = x.size(0)
        time_steps = x.size(1)
        
        out_shape = (batch_size, time_steps) + readout_states.shape[1:]
        mem_shape = (batch_size, time_steps) + mem.shape[1:]

        outputs = torch.empty(out_shape, dtype=readout_states.dtype, device=x.device)
        new_hidden_states = torch.empty(mem_shape, dtype=mem.dtype, device=x.device)
        new_readout_states = torch.empty(out_shape, dtype=readout_states.dtype, device=x.device)
        
        if self.encoding_in == ENCODING.DELTA:
            x = self.normalise(x, update=False, mean=means, std=stds, mask=mask)
            x = self.delta_modulator.encode(continuous_x, mask, current_thresholds)
        if self.encoding_in == ENCODING.RATE:
            x = self.normalise(x, update=False, mean=means, std=stds, mask=mask)
            x = torch.sigmoid(x)
            x = self.rate_encoder.encode(x, mask, current_thresholds[0])


        encoding_layer_spikes = []

        all_cur = self.fc1(x)  # Linear layer only processes the last dimension of the input tensor (i.e. only the feature dimension is used here)
        time_dim = 2 if self.encoding_in == ENCODING.RATE else 1
        for t in range(x.size(time_dim)):
            # Slice against time dimension - rate encoding has a virtual timesteps dimension, hence why it is done separately.
            if self.encoding_in == ENCODING.RATE:
                cur = all_cur[:, :, t, ...]  
            else:
                cur = all_cur[:, t, ...] 

            for vt in range(self.virtual_timesteps):
                if self.encoding_in == ENCODING.RATE:
                    cur_in = cur[vt, ...]
                else:
                    cur_in = cur

                for i in range(len(self.hidden_and_output_layers) - 1):
                    if i % 2 == 0:
                        mem_idx = i // 2
                        spk, new_mem = self.hidden_and_output_layers[i](cur_in, mem[..., mem_idx, :])
                        mem[..., mem_idx, :] = new_mem

                        if i == 0:
                            encoding_layer_spikes.append(spk)
                    else:
                        cur_in = self.hidden_and_output_layers[i](spk)

                _, new_readout = self.hidden_and_output_layers[-1](cur_in, readout_states)
                readout_states = new_readout

            self.current_encoding_rate = torch.stack(encoding_layer_spikes).mean()
            outputs[:, t, ...] = new_readout
            new_hidden_states[:, t, ...] = mem
            new_readout_states[:, t, ...] = new_readout

        new_prev_obs = continuous_x
        return outputs, mask, new_hidden_states, new_prev_obs, new_readout_states, current_thresholds

    def load_thresholds(self, path: str, batch_offset: int, agent_idx):
        """Load and update thresholds from file."""
        with open(path, 'r') as f:
            thresholds = json.load(f)
        # str() is needed as JSON requires keys to be strings.
        self.delta_modulator.thresholds = torch.tensor(thresholds[str(agent_idx)][-1], device=self.device)

    def load_scaling_factors(self, fp, agent_idx):
        with open(fp, 'r') as f:
            scaling_factors = json.load(f)
        self.rate_encoder.scaling_factors = torch.tensor(scaling_factors[str(agent_idx)][-1], device=self.device)

    
class AgentEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ("availableBW", c_double),
        ("averageQLength", c_double),
        ("txRateECN", c_double),
        ("k_min_in", c_double),
        ("k_delta_in", c_double),
        ("p_max_in", c_double)
    ]

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ("agents", (AgentEnv * NUM_PORTS) * NUM_AGENTS),
        ("activePorts", c_uint32 * NUM_AGENTS)
    ]

class AgentAct(Structure):
    _pack_ = 1
    _fields_ = [
        ("k_min_out", c_double),
        ("k_delta_out", c_double),
        ("p_max_out", c_double)
    ]

class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ("agents", (AgentAct * NUM_PORTS) * NUM_AGENTS)
    ]

class RLEnv(EnvBase):
    def __init__(self, sim_num, rl_interface, device="cpu"):
        self.batch_size = torch.Size([NUM_AGENTS])
        super().__init__(batch_size=self.batch_size, device=device)
        self.rl = rl_interface
        self.python2_path = "/home/links/rl624/.conda/envs/hpcc_env/bin/python"
        
        self.pool_id = 1234
        self.rl_id = 2333
        
        if TOPO == "STAR":
            topology = "60_incast"
            trace = "web_search_60_50"
        elif TOPO == "FAT":
            topology = "fat"
            trace = "web_search_fat_30"
        elif TOPO == "TESTING":
            topology = "star_5"
            trace = "web_search_5_80"
        else:
            raise Exception("TOPO must be either 'star' or 'fat'.")
        
        self.ns3_args = {
            "cc": "dcqcn",
            "bw": 100,  # NIC bandwidth
            "rl_ecn_marking": 1,
            "topo": topology, 
            "sim_num": sim_num,
            "encoding": args.enc_in.upper(),
            "firing_rate": FIRING_RATE
        }
        if PERFORM_EVAL:
            self.ns3_args["trace"] = f'{trace}_0.01s'
        else:
            self.ns3_args["trace"] = f'{trace}_100s'
        

        self.sim_done = True # Flag to determine whether to start the ns3 simulation.

        self.ns3_process = None

        self.max_active_ports = None

        # ============ FOR Debugging ==============
        self.w = 0.5
        self.alpha = 20
        self.reward_history = [{
            "rewards": [],
            "q_lengths": [],
            "throughput": [],
            "normalised_q_lengths": [],
            "normalised_throughput": [],
            "w": self.w,
            "alpha": self.alpha,
            "n": [],
            "T": [],
            "D": [],
            "k_min": [],
            "k_max": [],
            "p_max": []
        } for _ in range(NUM_AGENTS)]
    
    def _build_specs(self):
        self.observation_spec = Composite(
            observation=Unbounded(shape=(*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),
            hidden_states=Unbounded(
                shape=(*self.batch_size, self.max_active_ports, NUM_HIDDEN_LAYERS, NUM_HIDDEN_CELLS), 
                dtype=torch.float32,
                device=self.device
            ),
            prev_obs=Unbounded(
                shape=(*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS),
                dtype=torch.float32,
                device=self.device
            ),
            readout_states=Unbounded(
                shape=(*self.batch_size, self.max_active_ports, OUT_FEATURES),
                dtype=torch.float32,
                device=self.device
            ),
            shape=self.batch_size,
            device=self.device
        )

        if CONT_OR_DISC == "CONTINUOUS":
            low = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, self.max_active_ports, NUM_ACTIONS)
            high = torch.tensor([1_000_000.0, 1_000_000.0, 1.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, self.max_active_ports, NUM_ACTIONS)

            self.action_spec = Bounded(
                low=low,
                high=high,  
                shape=(*self.batch_size, self.max_active_ports, NUM_ACTIONS),
                dtype=torch.float32,
                device=self.device
            )
            max_k_min = high[0][0][0]
            max_k_delta = high[0][0][1]
            max_p_max = high[0][0][2]
        else:
            low = torch.tensor(0, dtype=torch.int64, device=self.device)
            high = torch.tensor(839, dtype=torch.int64, device=self.device)

            self.action_spec = Bounded(
                low=low,
                high=high,
                shape=(*self.batch_size, self.max_active_ports, 3), 
                dtype=torch.int64,
                device=self.device
            )
            alpha = 20
            max_k_min = alpha * 2^(ACTION_BINS[0]-1)
            max_k_delta = alpha * 2^(ACTION_BINS[1]-1)
            max_p_max = 1.0
        self.reward_spec = Unbounded(shape=(*self.batch_size, self.max_active_ports, 1), dtype=torch.float32, device=self.device)

        self.done_spec = Composite(
            done=Binary(1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            terminated=Binary(1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            truncated=Binary(1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            shape=self.batch_size,
            device=self.device
        )
        # Very important scaling to prevent exploding gradients, producing 'nan' actions.
        self.OBS_SCALE = torch.tensor([
            1e11/8.0,   # availableBW - normalised by BW
            1e8,    # averageQLength
            1e11,   # txRateECN
            max_k_min,
            max_k_delta,
            max_p_max         
        ], dtype=torch.float32, device=self.device)

        self.last_obs = torch.zeros(*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS, dtype=torch.float32, device=self.device)

    def set_mode(self, mode):
        print("in set_mode")
        if mode.upper() == "EVAL":
            print("SETTING TRACE")
            self.ns3_args["trace"] = "web_search_60_80_0.1s"
        else:
            pass
    
    def save_reward_history(self):
        with open(f"../reward_data_agent.txt", "w") as f:
            json.dump(self.reward_history, f)


    def start_ns3_simulation(self):
        """
        Start a new ns3 simulation. The previous simulation will be stopped before the new one is started.
        """
        # Remove previous shared memory corresponding to the current shared memory key.
        os.system(f"ipcrm -M {2333 + self.ns3_args['sim_num']} 2>/dev/null")
        os.system(f"ipcrm -S {2333 + self.ns3_args['sim_num']} 2>/dev/null")
        
        print(f"[Python] Initialising shmkey: {self.rl_id + self.ns3_args['sim_num']}")
        py_interface.Init(self.rl_id + self.ns3_args['sim_num'], 1040000) #131072)  # self.pool_id

        print("INITIALISED", flush=True)
        print(f"AgentEnv: {sizeof(AgentEnv)}")   
        print(f"AgentAct: {sizeof(AgentAct)}")   
        print(f"Env:      {sizeof(Env)}")       
        print(f"Act:      {sizeof(Act)}")        
        print(f"Total:    {sizeof(Env) + sizeof(Act)}")

        self.rl = py_interface.Ns3AIRL(self.rl_id + self.ns3_args['sim_num'], Env, Act)

        env = os.environ.copy()
        env["PATH"] = f"{os.path.dirname(self.python2_path)}:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

        for key in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "PYTHON"]:
            env.pop(key, None)

        args_str = [f'--{key}={value}' for key, value in self.ns3_args.items()]
        if PERFORM_EVAL:
            args_str.append("--perform_eval")

        self.ns3_process = subprocess.Popen([self.python2_path, "run.py", *args_str], env=env, text=True)

    
    def get_obs(self, data):
        new_obs = []
        for agent_num in range(NUM_AGENTS):
            agent_data = []
            for port_num in range(self.max_active_ports):
                agent_data.append([            
                    data.env.agents[agent_num][port_num].availableBW, 
                    data.env.agents[agent_num][port_num].averageQLength,
                    # data.env.agents[agent_num][port_num].maxQLength,
                    data.env.agents[agent_num][port_num].txRateECN,
                    data.env.agents[agent_num][port_num].k_min_in,
                    data.env.agents[agent_num][port_num].k_delta_in,
                    data.env.agents[agent_num][port_num].p_max_in
                ])
            new_obs.append(agent_data)

        new_obs = torch.tensor(new_obs, device=self.device, dtype=torch.float32)
        return new_obs / self.OBS_SCALE


    def _step(self, td):
        actions = td["action"]

        with self.rl as data:
            if data is None:
                
                self.sim_done = True
                return TensorDict({
                    "observation": self.last_obs,
                    "reward": torch.zeros(*self.batch_size, self.max_active_ports, 1, dtype=torch.float32, device=self.device),
                    "done": torch.ones(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                    "terminated": torch.ones(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                    "truncated": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
                }, batch_size=self.batch_size, device=self.device)
            else:

                for i in range(NUM_AGENTS):
                    for j in range(self.max_active_ports):
                        if CONT_OR_DISC == "CONTINUOUS":
                            # The actions are simply the thresholds to use
                            data.act.agents[i][j].k_min_out = actions[i, j, 0].item()
                            data.act.agents[i][j].k_delta_out = actions[i, j, 1].item()
                            data.act.agents[i][j].p_max_out = actions[i, j, 2].item()
                        else:
                            k_min_idx = actions[i, j, 0].item()
                            k_delta_idx = actions[i, j, 1].item()
                            p_max_idx = actions[i, j, 2].item()


                            alpha = 20 
                            # N.b. multiplied by 1024 as ACC uses KB as their queue length metric.
                            k_min_val = alpha * (2.0 ** k_min_idx) * 1024.0 
                            k_delta_val = alpha * (2.0 ** k_delta_idx) * 1024.0
                            p_max_val = p_max_idx / 20.0

                            data.act.agents[i][j].k_min_out = k_min_val
                            data.act.agents[i][j].k_delta_out = k_delta_val
                            data.act.agents[i][j].p_max_out = p_max_val

                new_obs = self.get_obs(data)
                reward = self._calculate_reward(data.env).reshape(*self.batch_size, self.max_active_ports, 1)

                self.last_obs = new_obs

            truncated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
            terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
            done = truncated | terminated 

            return TensorDict({
                "observation": new_obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            }, batch_size=self.batch_size, device=self.device)

    def _reset(self, tensordict: TensorDictBase | None = None):
        
        if self.sim_done:
            print("CALLING start_ns3_simulation", flush=True)
            self.start_ns3_simulation()

            print("STARTED ns3 simulation", flush=True)
            
            new_obs = None
            count =0 # FOR DEBUGGING
            while new_obs is None:
                if self.ns3_process.poll() is not None:
                    raise RuntimeError(f"ns-3 process died unexpectedly...")
                if count % 100 == 0:
                    print(f"STUCK in received first obs: count={count}, isFinish: {self.rl.isFinish()}", flush=True)
                with self.rl as data:  # Calls rl.Acquire() and rl.ReleaseMemory() on __enter__ and __exit__ respectively.
                    
                    
                    if data is not None:
                        cur_max_active_ports = max(list(data.env.activePorts))
                        if cur_max_active_ports == 0:
                            continue
                        else:
                            if self.max_active_ports is None:
                                self.max_active_ports = cur_max_active_ports

                                self.valid_mask = torch.zeros((NUM_AGENTS, self.max_active_ports), dtype=torch.bool, device=self.device)
                                for i, active_count in enumerate(data.env.activePorts):
                                    self.valid_mask[i, :active_count] = True
                                
                                self._build_specs()

                            new_obs = self.get_obs(data)

                            print("RECEIVED FIRST obs", flush=True)
                    else:
                        print(f"Data is None! isFinish={self.rl.isFinish()}", flush=True)
                count += 1
                time.sleep(0.01)

            print("Control back to python", flush=True)
            self.sim_done = False
        else:
            new_obs = self.last_obs
        
       
        res = TensorDict(
            {
                "observation": new_obs,
                "mask": self.valid_mask.unsqueeze(-1),
                "hidden_states": torch.zeros((*self.batch_size, self.max_active_ports, NUM_HIDDEN_LAYERS, NUM_HIDDEN_CELLS), dtype=torch.float32, device=self.device),
                "prev_obs": torch.zeros((*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),  # N.B. this is only used for delta modulation (temporal encoding) input.
                "readout_states": torch.zeros((*self.batch_size, self.max_active_ports, OUT_FEATURES), dtype=torch.float32, device=self.device),
                "delta_thresholds": torch.zeros((*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),
                "means": torch.zeros((*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),
                "stds": torch.ones((*self.batch_size, self.max_active_ports, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device)
            },
            batch_size=self.batch_size,
            device=self.device
        )

        return res

    def _set_seed(self, seed: int | None = None, static_seed: bool = False):
        torch.manual_seed(seed)
        return seed + 1

    def _calculate_reward(self, env_data):
        rewards = []
        for i in range(NUM_AGENTS):
            port_rewards = []
            for j in range(self.max_active_ports):
                availableBW = env_data.agents[i][j].availableBW
                avg_q_len = env_data.agents[i][j].averageQLength

                # Calculate min n, such that E(n) > L, E(n) = a * (2**n)
                n_power = 0
                for n in range(0, 10):
                    n_power = n
                    # N.B. divide by 1000 as alpha is set in the ACC paper based on the queue length in KB but avg_q_len is in bytes here.
                    if self.alpha * 2**n > (avg_q_len/1000):
                        break

                D = 1 - n_power / 10

                T = -availableBW / (100_000_000_000 / 8.0)

                w = 0.5
                r = w * T +(1 - w) * D

                port_rewards.append(r)
            rewards.append(port_rewards)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        return rewards

class MultiAgentPolicyWrapper(nn.Module):
    def __init__(self, local_policies, use_mappo, num_agents):
        super().__init__()
        self.use_mappo = use_mappo
        self.num_agents = num_agents
        self.policies = nn.ModuleList(local_policies)

    def forward(self, tensordict):
        output_tds = []
        
        for i in range(self.num_agents):
            policy_idx = i# 0 if self.use_mappo else i
            out_td = self.policies[policy_idx](tensordict[i])
            output_tds.append(out_td)
            
        batched_output = torch.stack(output_tds, dim=0)
        tensordict.update(batched_output)
        
        return tensordict

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
env = RLEnv(SIM_NUM, None, device=device)
initial_td = env.reset()
saved_models_suffix = f"{TOPO}_{'snn' if USE_SNN else 'ann'}_{CONT_OR_DISC}_{NUM_HIDDEN_LAYERS}_hidden_layers_{NUM_HIDDEN_CELLS}_hidden_cells_{INPUT_ENCODING.name}_{OUTPUT_ENCODING.name}_BETA_{SNN_BETA}_FIRING_RATE_{FIRING_RATE}"
if INPUT_ENCODING == ENCODING.DIRECT:
    saved_models_suffix += f"_MSE_SCALING_{MSE_LOSS_SCALING}"
# elif INPUT_ENCODING == ENCODING.DELTA:
#     saved_models_suffix += f"_telem_momentum_{MOMENTUM}"

def make_policy_module(agent_idx):
    actor_in_keys = ["observation"]
    if USE_SNN:
        # print("Using SNN", flush=True)
        actor_in_keys = ["observation", "mask", "hidden_states", "prev_obs", "readout_states", "delta_thresholds", "means", "stds"]
        actor_out_keys = ["logits" if CONT_OR_DISC == "DISCRETE" else "raw_output", ("next", "mask"), ("next", "hidden_states"), ("next", "prev_obs"), ("next", "readout_states"),
                          ("next", "delta_thresholds"), ("next", "means"), ("next", "stds")]
        
        actor_net = SNNNetwork(
            num_hidden_layers = NUM_HIDDEN_LAYERS, 
            num_hidden = NUM_HIDDEN_CELLS, 
            out_features = 2 * NUM_ACTIONS if CONT_OR_DISC == "CONTINUOUS" else OUT_FEATURES, 
            encoding_in = INPUT_ENCODING, 
            encoding_out = OUTPUT_ENCODING, 
            beta = SNN_BETA, 
            saved_models_suffix = saved_models_suffix, 
            virtual_timesteps = 20, 
            device=device,
            debug_eval=False
        )                
    else:
        actor_in_keys = ["observation"]
        actor_out_keys = ["logits" if CONT_OR_DISC == "DISCRETE" else "raw_output"]

        hidden_layers = []
        for _ in range(NUM_HIDDEN_LAYERS):
            hidden_layers.append(nn.Linear(NUM_HIDDEN_CELLS, NUM_HIDDEN_CELLS))
            hidden_layers.append(nn.ReLU())

        actor_net = nn.Sequential(
            nn.Linear(NUM_STATE_PARAMS, NUM_HIDDEN_CELLS),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(NUM_HIDDEN_CELLS, 2 * NUM_ACTIONS if CONT_OR_DISC == "CONTINUOUS" else OUT_FEATURES)
        )

    if CONT_OR_DISC == "CONTINUOUS":
        actor_module = TensorDictModule(
            actor_net,
            in_keys=actor_in_keys,
            out_keys=actor_out_keys
        )

        extraction_module = TensorDictModule(
            NormalParamExtractor(),
            in_keys=["raw_output"],
            out_keys=["loc", "scale"]
        )

        policy_module = TensorDictSequential(
            actor_module,
            extraction_module
        )
    else:
        actor_module = TensorDictModule(
            actor_net,
            in_keys=actor_in_keys,
            out_keys=actor_out_keys
        )
        policy_module = actor_module

    return policy_module

def build_global_state(obs, mask):
    masked_obs = obs * mask
    target_shape = list(obs.shape)
    target_shape[-1] = -1
    
    if obs.dim() == 3: 
        universe = masked_obs.flatten()
        return universe.unsqueeze(0).expand(NUM_AGENTS, -1)
    elif obs.dim() == 4: 
        time_steps = obs.shape[1]
        obs_time_first = masked_obs.permute(1, 0, 2, 3)
        universe_per_time = obs_time_first.flatten(start_dim=1)
        return universe_per_time.unsqueeze(0).expand(NUM_AGENTS, time_steps, -1)
    else:
        raise ValueError(f"Unexpected observation dimension: {obs.shape}")

class MAPPOValueNetwork(nn.Module):
    def __init__(self, critic_input_dim, num_hidden_cells):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(critic_input_dim, num_hidden_cells),
            nn.Tanh(),
            nn.Linear(num_hidden_cells, num_hidden_cells),
            nn.Tanh(),
            nn.Linear(num_hidden_cells, 1)
        )

    def forward(self, local_obs, raw_global_state):
        num_ports = local_obs.shape[-2]
        expanded_global = raw_global_state.unsqueeze(-2)
        target_shape = list(raw_global_state.shape)
        target_shape.insert(-1, num_ports)
        expanded_global = expanded_global.expand(*target_shape)
        critic_in = torch.cat([local_obs, expanded_global], dim=-1)
        return self.mlp(critic_in)

def make_value_module(centralised, max_active_ports):
    if centralised:
        input_dim = NUM_STATE_PARAMS + NUM_STATE_PARAMS * NUM_AGENTS * max_active_ports
        in_keys = ["observation", "raw_global_state"]
        value_net = MAPPOValueNetwork(input_dim, NUM_HIDDEN_CELLS)
    else:
        input_dim = NUM_STATE_PARAMS
        in_keys=["observation"]


        value_net = nn.Sequential(
            nn.Linear(input_dim, NUM_HIDDEN_CELLS),
            nn.Tanh(),
            nn.Linear(NUM_HIDDEN_CELLS, NUM_HIDDEN_CELLS),
            nn.Tanh(),
            nn.Linear(NUM_HIDDEN_CELLS, 1)
        )

    value_module = ValueOperator(
        module=value_net,
        in_keys=in_keys,
    )

    return value_module


def get_snn_net(policy):
    '''
    Utility function to make it easier if the structure of the ProbabilisticActor modules change.
    '''
    if CONT_OR_DISC == "CONTINUOUS":
        return policy.module[0].module[0]
    else:
        return policy.module[0].module

if CONT_OR_DISC == "CONTINUOUS":
    local_policies = [ProbabilisticActor(
        module=make_policy_module(i),
        spec=env.action_spec[0],
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={ 
            "low": env.action_spec[0].space.low, 
            "high": env.action_spec[0].space.high,
        },
        return_log_prob=True,
    ) for i in range(NUM_AGENTS)]
else:
    local_policies = [ProbabilisticActor(
        module=make_policy_module(i),
        spec=env.action_spec[0],
        in_keys=["logits"],
        distribution_class=SplitMultiCategorical, # Categorical,
        
        return_log_prob=True,
    ) for i in range(NUM_AGENTS)]

# If training was interrupted use this to load the previous models and continue the training.
if BATCH_OFFSET != 0 or PERFORM_EVAL:
    for agent_num in range(NUM_AGENTS):
        print(f"Loading policy: {agent_num}", flush=True)
        if USE_MAPPO:
            # All agents use the same learnt policy.
            loaded_state_dict = torch.load(f'../saved_models_{saved_models_suffix}/agent_0_batch_{BATCH_OFFSET}_policy.pth')
        else:
            loaded_state_dict = torch.load(f'../saved_models_{saved_models_suffix}/agent_{agent_num}_batch_{BATCH_OFFSET}_policy.pth')
        local_policies[agent_num].load_state_dict(loaded_state_dict)

        if USE_SNN:
            snn_net = get_snn_net(local_policies[agent_num])
            if INPUT_ENCODING == ENCODING.DELTA:
                print(f'Loading agent {agent_num} thresholds', flush=True)
                snn_net.load_thresholds(f'../saved_models_{saved_models_suffix}/delta_thresholds.json', BATCH_OFFSET, agent_num)
            elif INPUT_ENCODING == ENCODING.RATE:
                print(f'Loading agent {agent_num} scaling factors', flush=True)
                snn_net.load_scaling_factors(f'../saved_models_{saved_models_suffix}/scaling_factors.json', agent_num)
            

multi_agent_policy_module = MultiAgentPolicyWrapper(local_policies, use_mappo=USE_MAPPO, num_agents=NUM_AGENTS) # So each agent's policy can be executed in the same environment step.

multi_agent_policy_module = multi_agent_policy_module.to(device)
multi_agent_policy_module(initial_td)

collector = SyncDataCollector(
    env,
    multi_agent_policy_module,
    frames_per_batch=NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH * NUM_AGENTS,
    total_frames=TOTAL_FRAMES,
    split_trajs=False,
    device=device,
)

lr = 1e-3 
max_grad_norm = 1.0
gamma = 0.99
lmbda = 0.95


samplers = [SliceSampler(
    slice_len=SEQUENCE_LENGTH,
    strict_length=True,
    cache_values=True,
    end_key=("next", "done"),
    traj_key=("collector", "traj_ids"),  # Differentiate between samples with the trajectory id.
) for _ in range(NUM_AGENTS)]

replay_buffers = [TensorDictReplayBuffer(
    storage=LazyMemmapStorage(max_size=10000),
    sampler=samplers[i],
    batch_size=(NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH) // NUM_MINI_BATCHES, # i.e. num_frames_per_batch / num_mini_batches = num_frames_per_mini_batch
) for i in range(NUM_AGENTS)]

entropy_eps = 0.001
clip_epsilon = (
    0.2  
)

loss_critic_coeff = 1.0
loss_entropy_coeff = entropy_eps

with torch.no_grad():
    global_state = build_global_state(initial_td["observation"], initial_td["mask"])
    print(f"Expanded global state: {global_state.shape}", flush=True)
    initial_td['raw_global_state'] = global_state


value_modules = [make_value_module(USE_MAPPO, env.max_active_ports) for _ in range(NUM_NETWORKS)]
for i in range(NUM_NETWORKS):
    value_modules[i] = value_modules[i].to(device)
    value_modules[i](initial_td)


advantage_modules = [GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_modules[i], average_gae=True, device=device,
) for i in range(NUM_NETWORKS)]

loss_modules = [ClipPPOLoss(
    actor_network=local_policies[i],
    critic_network=value_modules[0],
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    critic_coeff=1.0,
    loss_critic_type="smooth_l1",
    reduction="none",
    normalize_advantage=False
) for i in range(NUM_AGENTS)]

optims = [torch.optim.Adam(loss_modules[i].parameters(), lr) for i in range(NUM_AGENTS)]

BATCH_LIMIT = 150

total_training_frames = BATCH_LIMIT * NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
    optims[i], total_training_frames // (NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH), 0.0
) for i in range(NUM_AGENTS)]

logs = defaultdict(list)
pbar = tqdm(total=TOTAL_FRAMES)
eval_str = ""

delta_modulation_thresholds = {}
rate_encoding_scaling_factors = {}
def align_input_tensors(td):
    """
    Make the done states have the same shape as the value and reward tensors. This is because
    the environment only uses the switch for the done signal, whereas the reward is on
    a per-port basis.
    """
    reward = td["next", "reward"]
    td["next", "done"] = td["next", "done"].unsqueeze(-1).expand_as(reward)
    td["next", "terminated"] = td["next", "terminated"].unsqueeze(-1).expand_as(reward)
    td["next", "truncated"] = td["next", "truncated"].unsqueeze(-1).expand_as(reward)

    return td

# Reset average firing rates from env.reset()
for i in range(NUM_AGENTS):
    snn_net = get_snn_net(local_policies[i])
    snn_net.reset_firing_rates()


avg_firing_rates = []
# DO TRAINING!
if not PERFORM_EVAL:
    # Check if 'saved_models' directory exists.
    if os.path.isdir(f'../saved_models_{saved_models_suffix}') is False:
        os.mkdir(f'../saved_models_{saved_models_suffix}')
    else:
        raise Exception(f'folder saved_models_{saved_models_suffix} already exists! Delete Manually if desired.')

    NUM_EPOCHS = 10
    tracked_data = {agent_num: {} for agent_num in range(NUM_AGENTS)}
    print("REACHED LOOP", flush=True)

    loss_values = []

    for i, tensordict_data in enumerate(collector):
        print("Batch collected", flush=True)
        # print(tensordict_data)
        batch_num = i + BATCH_OFFSET  # TEMPORARY: interrupted training
        if i > BATCH_LIMIT:
            print("BATCH_LIMIT reached - finishing now!", flush=True)
            break
        
        all_agents_data = []
        
        if USE_MAPPO:
            with torch.no_grad():
                tensordict_data["raw_global_state"]= build_global_state(tensordict_data["observation"], tensordict_data["mask"])
                tensordict_data["next", "raw_global_state"] = build_global_state(tensordict_data["next", "observation"], tensordict_data["next", "mask"])
            
                for agent_num in range(NUM_AGENTS):
                    agent_data = tensordict_data[agent_num]
                    agent_data = agent_data.view(NUM_DESIRED_SEQUENCES, SEQUENCE_LENGTH)
                    agent_data = align_input_tensors(agent_data)
                    
                    advantage_modules[0](agent_data) 
                    all_agents_data.append(agent_data)

            combined_data = torch.cat(all_agents_data, dim=0)
            total_sequences = NUM_AGENTS * NUM_DESIRED_SEQUENCES
            sequences_per_minibatch = (NUM_AGENTS * NUM_DESIRED_SEQUENCES) // NUM_MINI_BATCHES

        cur_loss_values = {}
        for agent_num in range(NUM_AGENTS):
            if not USE_MAPPO:  # IPPO
                agent_data = tensordict_data[agent_num]

                agent_data = align_input_tensors(agent_data)
                with torch.no_grad():
                    advantage_modules[agent_num](agent_data)

                replay_buffers[agent_num].extend(agent_data)

            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            update_steps = 0

            for epoch in range(NUM_EPOCHS):
                print(f"Agent: {agent_num}, Batch {i + BATCH_OFFSET}, epoch {epoch}", flush=True)
                if USE_MAPPO:
                    perm = torch.randperm(total_sequences)
                for mini_batch_idx in range(NUM_MINI_BATCHES): 
                    if USE_MAPPO:
                        batch_indices = perm[mini_batch_idx * sequences_per_minibatch : (mini_batch_idx + 1) * sequences_per_minibatch]
                        subdata = combined_data[batch_indices]  # Slice along sequence dimension so timesteps are still sequential.
                    else:
                        subdata = replay_buffers[agent_num].sample()
                        subdata = align_input_tensors(subdata)
                        sequences_per_minibatch = NUM_DESIRED_SEQUENCES // NUM_MINI_BATCHES
                        subdata = subdata.view(sequences_per_minibatch, SEQUENCE_LENGTH)

                    
                    loss_vals = loss_modules[agent_num](subdata.to(device))
                    mask=subdata["mask"].squeeze(-1).to(device)  # N.B. this mask only matters when there is more than 1 switch and there are a different number of active ports between them.

                    masked_policy_loss = (loss_vals["loss_objective"] * mask).sum() / mask.sum()
                    masked_value_loss = (loss_vals["loss_critic"] * mask).sum() / mask.sum()
                    masked_entropy = (loss_vals["loss_entropy"] * mask).sum() / mask.sum()

                    total_loss = masked_policy_loss + masked_value_loss + masked_entropy

                    if INPUT_ENCODING == ENCODING.DIRECT:
                        snn_net = get_snn_net(local_policies[agent_num])
                        actual_rate = snn_net.current_encoding_rate 

                        target_rate = torch.tensor(FIRING_RATE, device=device)

                        rate_penalty = F.mse_loss(actual_rate, target_rate) 
                        total_loss += MSE_LOSS_SCALING * rate_penalty

                    if torch.isnan(total_loss):
                        print(f"NaN loss at batch {batch_num}, epoch {epoch}, agent {agent_num}", flush=True)
                        print(f"  loss_objective: {loss_vals['loss_objective']}", flush=True)
                        print(f"  loss_critic: {loss_vals['loss_critic']}", flush=True)
                        print(f"  loss_entropy: {loss_vals['loss_entropy']}", flush=True)

                    

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(loss_modules[agent_num].parameters(), max_grad_norm)
                    optims[agent_num].step()
                    optims[agent_num].zero_grad()

                    total_policy_loss += masked_policy_loss.item()
                    total_value_loss += masked_value_loss.item()
                    total_entropy_loss += masked_entropy.item()

                    update_steps += 1

            avg_policy_loss = total_policy_loss / update_steps
            avg_value_loss = total_value_loss / update_steps
            avg_entropy_loss = total_entropy_loss / update_steps
            cur_loss_values[agent_num] = [avg_policy_loss, avg_value_loss, avg_entropy_loss]

            replay_buffers[agent_num].empty()
            schedulers[agent_num].step()
        loss_values.append(cur_loss_values)
        collector.update_policy_weights_()
           
        if USE_SNN:
            if INPUT_ENCODING == ENCODING.DIRECT:
                snn_net = get_snn_net(local_policies[0])
                actual_rate = snn_net.current_encoding_rate 
                print("THE ACTUAL RATE IS: ", actual_rate)
                avg_firing_rates.append(actual_rate.cpu().detach().item())
                
            elif INPUT_ENCODING == ENCODING.DELTA:
                snn_net = get_snn_net(local_policies[0])
                firing_rate_hist = snn_net.delta_modulator.spike_rate_hist
                avg_firing_rates = firing_rate_hist
            elif INPUT_ENCODING == ENCODING.RATE:
                snn_net = get_snn_net(local_policies[0])
                firing_rate_hist = snn_net.rate_encoder.spike_rate_hist
                avg_firing_rates = firing_rate_hist

        if batch_num % 10 == 0:
            fig, ax = plt.subplots()

            # Indexed by [Element][Agent][Loss_Idx]
            policy_loss = [loss_values[i][0][0] for i in range(len(loss_values))]
            value_loss = [loss_values[i][0][1] for i in range(len(loss_values))]
            entropy_loss = [loss_values[i][0][2] for i in range(len(loss_values))]
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(policy_loss)
            fig.savefig(f'../losses/policy_loss_{saved_models_suffix}.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(value_loss)
            fig.savefig(f'../losses/value_loss_{saved_models_suffix}.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(entropy_loss)
            fig.savefig(f'../losses/entropy_loss_{saved_models_suffix}.png')
            plt.close(fig)

            with open(f'../saved_models_{saved_models_suffix}/loss_values.json', 'w') as f:
                json.dump(loss_values, f)
                
            for agent_num in range(NUM_NETWORKS):
                torch.save(local_policies[agent_num].state_dict(), f'../saved_models_{saved_models_suffix}/agent_{agent_num}_batch_{i}_policy.pth')
            
            if USE_SNN:
                if INPUT_ENCODING == ENCODING.DELTA:
                    for agent_idx in range(NUM_AGENTS):
                        snn_net = get_snn_net(local_policies[agent_idx])
                        
                        hist = snn_net.delta_modulator.threshold_hist
                        hist = [x.cpu().detach().tolist() for x in hist]

                        delta_modulation_thresholds[agent_idx] = hist
                    
                    with open(f'../saved_models_{saved_models_suffix}/delta_thresholds.json', 'w') as f:
                        json.dump(delta_modulation_thresholds, f)
                    
                    colours = ['b', 'r', 'g', 'orange', 'purple', 'teal', 'gold']
                    fig, ax = plt.subplots()
                    for i in range(NUM_STATE_PARAMS):
                        ax.plot([x.cpu()[i] for x in avg_firing_rates], label=f"Firing Rate {i}", color=colours[i])
                    
                    ax.set_xlabel("Threshold Updates")
                    ax.set_ylabel("Firing Rate")
                    fig.legend()
                    fig.tight_layout()
                    fig.savefig(f'../firing_rates/{saved_models_suffix}_individual.png')
                    plt.close(fig)

                    fig, ax = plt.subplots()

                    ax.plot([sum(x.cpu()) / len(x.cpu()) for x in avg_firing_rates], label=f"Firing Rate", color='r')
                    
                    ax.set_xlabel("Threshold Updates")
                    ax.set_ylabel("Average Firing Rate")
                    fig.legend()
                    fig.tight_layout()
                    fig.savefig(f'../firing_rates/{saved_models_suffix}_collective.png')
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    for i in range(NUM_STATE_PARAMS):
                        ax.plot([x[i] for x in delta_modulation_thresholds[0]], label=f"Threshold {i}", color=colours[i])
                    
                    ax.set_xlabel("Threshold Updates")
                    ax.set_ylabel("Threshold Value")

                    fig.legend()
                    fig.tight_layout()
                    fig.savefig(f'../firing_rates/{saved_models_suffix}_thresholds.png')
                    plt.close(fig)


                    avg_firing_rates = [x.cpu().detach().tolist() for x in avg_firing_rates]
                elif INPUT_ENCODING == ENCODING.RATE:
                    for agent_idx in range(NUM_AGENTS):
                        snn_net = get_snn_net(local_policies[agent_idx])
                        
                        hist = snn_net.rate_encoder.scaling_factor_hist
                        hist = [x.cpu().detach().tolist() for x in hist]

                        rate_encoding_scaling_factors[agent_idx] = hist

                    with open(f'../saved_models_{saved_models_suffix}/scaling_factors.json', 'w') as f:
                        json.dump(rate_encoding_scaling_factors, f)

                    fig, ax = plt.subplots()

                    colours = ['b', 'r', 'g', 'orange', 'purple', 'teal', 'gold']
                    for i in range(NUM_STATE_PARAMS):
                        ax.plot([x.cpu()[i] for x in avg_firing_rates], label=f"Firing Rate {i}", color=colours[i])
                    
                    ax.set_xlabel("Scaling Factor Updates")
                    ax.set_ylabel("Firing Rate")
                    fig.legend()
                    fig.tight_layout()
                    fig.savefig(f'../firing_rates/{saved_models_suffix}_individual_rates.png')
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    for i in range(NUM_STATE_PARAMS):
                        ax.plot([x[i] for x in rate_encoding_scaling_factors[0]], label=f"Scaling Factor {i}", color=colours[i])
                    
                    ax.set_xlabel("Scaling Factor Updates")
                    ax.set_ylabel("Scaling Factor")
                    fig.legend()
                    fig.tight_layout()
                    fig.savefig(f'../firing_rates/{saved_models_suffix}_individual_scaling_factors.png')
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    ax2 = ax.twinx()

                    ax.plot([sum(x.cpu()) / len(x.cpu()) for x in avg_firing_rates], label=f"Average Firing Rate", color='r')
                    
                    # Agent 0
                    for i in range(NUM_STATE_PARAMS):
                        ax2.plot([x[i] for x in rate_encoding_scaling_factors[0]],label=f"Scaling Factor {i}", color='b')
                    
                    ax.set_xlabel("Scaling Factor Updates")
                    ax.set_ylabel("Firing Rate")
                    ax2.set_ylabel("Scaling Factor")
                    fig.legend()
                    fig.tight_layout()
                    fig.savefig(f'../firing_rates/{saved_models_suffix}_singular.png')
                    plt.close(fig)

                    avg_firing_rates = [x.cpu().detach().tolist() for x in avg_firing_rates]
                elif INPUT_ENCODING == ENCODING.DIRECT:
                    fig, ax = plt.subplots()
                    print(f"Agent 0 firing rate: {avg_firing_rates[-1]}")
                    ax.plot(list(range(len(avg_firing_rates))), avg_firing_rates)
                    fig.savefig(f'../firing_rates/direct_encoding_{saved_models_suffix}.png')
                    plt.close(fig)
                    # This is the firing rates of the encoding layer for direct encoding.
                    # The firing rate before the input layer for delta and rate encoding.
                with open(f'../saved_models_{saved_models_suffix}/firing_rates.json', 'w') as f:
                    json.dump(avg_firing_rates, f)

else:
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        print("RUNNING EVAL ROLLOUT", flush=True)

        if USE_SNN and INPUT_ENCODING == ENCODING.DIRECT:
            for agent_idx in range(NUM_AGENTS):
                snn_net = get_snn_net(local_policies[agent_idx])
                FIRST_LAYERS_SPIKING_AVERAGE = 0
                FIRST_LAYERS_SPIKING_NUM_SAMPLES = 0
        elif USE_SNN and INPUT_ENCODING == ENCODING.RATE:
            for agent_idx in range(NUM_AGENTS):
                snn_net = get_snn_net(local_policies[agent_idx])
                snn_net.rate_encoder.reset_firing_averages()
        elif USE_SNN and INPUT_ENCODING == ENCODING.DELTA:
            for agent_idx in range(NUM_AGENTS):
                snn_net = get_snn_net(local_policies[agent_idx])
                snn_net.delta_modulator.reset_firing_averages()

        eval_rollout = env.rollout(10_000_000, multi_agent_policy_module) # Run until the simulation ends
        print("DONE ROLLOUT", flush=True)
        env.save_reward_history()

        if USE_SNN and INPUT_ENCODING == ENCODING.DIRECT:
            for agent_idx in range(NUM_AGENTS):
                snn_net = get_snn_net(local_policies[agent_idx])
                print(f"Agent {agent_idx} Overall Eval Firing Rate: {snn_net.current_encoding_rate}")
                print(f"avg: {FIRST_LAYERS_SPIKING_AVERAGE}, steps: {FIRST_LAYERS_SPIKING_NUM_SAMPLES}", flush=True)
                fig, ax = plt.subplots()
                ax.plot(snn_net.firing_rate_hist.cpu())

                fig.savefig("../eval_logs/0.85_firing_rate.png")
                plt.close(fig)
        elif USE_SNN and INPUT_ENCODING == ENCODING.RATE:
            for agent_idx in range(NUM_AGENTS):
                snn_net = get_snn_net(local_policies[agent_idx])
                print(f"Num steps: {snn_net.step_count}, active_ports: {env.max_active_ports}", flush=True)
                firing_avgs = snn_net.rate_encoder.get_current_firing_averages(env.max_active_ports, snn_net.step_count, debug=True)
                print(f"Agent {agent_idx} Overall Eval Firing Rate: {firing_avgs}", flush=True)
        elif USE_SNN and INPUT_ENCODING == ENCODING.DELTA:
            for agent_idx in range(NUM_AGENTS):
                snn_net = get_snn_net(local_policies[agent_idx])
                print(f"Num steps: {snn_net.step_count}, active_ports: {env.max_active_ports}", flush=True)
                firing_avgs = snn_net.delta_modulator.get_current_firing_averages(env.max_active_ports, snn_net.step_count, debug=True)
                print(f"Agent {agent_idx} Overall Eval Firing Rate: {firing_avgs}", flush=True)
