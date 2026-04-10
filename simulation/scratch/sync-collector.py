import sys
sys.dont_write_bytecode = True  # Prevent __pycache__ directory being created.

from collections import defaultdict

from tensordict import TensorDict, TensorDictBase
import torch
import torch.multiprocessing as mp
import py_interface
from ctypes import Structure, c_double, c_uint64, c_uint32
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
from tqdm import tqdm
import subprocess
import os
import json
import signal
import psutil
import time
import gc
from snntorch import spikegen
# from torchinfo import summary
import argparse
import matplotlib.pyplot as plt

from delta_modulator import DeltaModulator

class OBS(Enum):
    BW = 0
    txRate = 1
    averageQLength = 2
    txRateECN = 3
    k_min_in = 4
    k_delta_in = 5
    p_max_in = 6

class ENCODING(Enum):
    DIRECT = 0
    DELTA = 1

# Create parser with description
parser = argparse.ArgumentParser(description="RL parser")

# TODO: use these arguments to name the saved_models directory and add a appropriate suffix.
parser.add_argument("--network_type", type=str, default="ann", help="ANN or SNN")
parser.add_argument("--eval", action="store_true", help="Eval mode")
parser.add_argument("--batch_offset", type=int, default=0, help="Batch offset for eval")
parser.add_argument("--hidden_layers", type=int, default=1, help="Num hidden layers")
parser.add_argument("--hidden_neurons", type=int, default=64, help="Num hidden cells")
parser.add_argument("--enc_in", type=str, default="direct", help="Encoding in - only for snn")
parser.add_argument("--enc_out", type=str, default="direct", help="Encoding out - only for snn")

# Optional argument
parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed output")

args = parser.parse_args()

BATCH_OFFSET = args.batch_offset
NUM_HIDDEN_LAYERS = args.hidden_layers
NUM_HIDDEN_CELLS = args.hidden_neurons
INPUT_ENCODING = ENCODING[args.enc_in.upper()]
OUTPUT_ENCODING = ENCODING[args.enc_out.upper()]
PERFORM_EVAL = args.eval


if args.network_type.lower() == "snn":
    USE_SNN = True
else:
    USE_SNN = False

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


# raise Exception("TESTING")

# TODO: align OUT_FEATURES and NUM_ACTIONS so that the output layers is only dependent on OUT_FEATURES (as hybrid encoding approaches may be taken).
NUM_ACTIONS = 3
OUT_FEATURES = 6
NUM_STATE_PARAMS = 8 # 7

NUM_AGENTS = 60 # 640 for 320 host fat tree
# FRAMES_PER_BATCH = 1024
TOTAL_FRAMES = 12_000_000
SEQUENCE_LENGTH = 16

# N.B. NUM_DESIRED_SEQUENCES must be divisible by NUM_MINI_BATCHES.
NUM_DESIRED_SEQUENCES = 64
NUM_MINI_BATCHES = 4

if NUM_DESIRED_SEQUENCES % NUM_MINI_BATCHES != 0:
    raise Exception("NUM_DESIRED_SEQUENCES must be divisible by NUM_MINI_BATCHES.")

class SNNNetwork(nn.Module):
    # TODO: Add a batch dimension so that parallel training is possible.
    def __init__(self, num_hidden_layers, num_hidden, out_features, encoding_in, encoding_out, debug_eval = False):
        super().__init__()

        if encoding_out != ENCODING.DIRECT:
            raise Exception("Only Direct Encoding is currently supported for the output.")

        self.encoding_in = encoding_in  # Enum
        
        # Set a dummy threshold - will adjust to more reasonable values after the first training run.
        init_thresholds = torch.tensor([0.0000, 0.2759, 0.0673, 0.7518, 0.9891, 0.4962, 0.2], device='cuda:0')
        
        spike_density = 0.2

        self.delta_modulator = DeltaModulator(
            init_thresholds=init_thresholds,
            val_names=["bw","txRate","averageQLength","maxQLength","txRateECN","k_min","k_delta","p_max"],
            learning_rate=0.1,
            window_size=2000,  # Approximately update thresholds once every other batch.
            spike_density=spike_density,
            fix_thresholds=True
        )

        # initialize layers
        self.num_hidden = num_hidden
        BETA = 0.8  # N.B. to make this a learnable parameter, wrap in nn.Parameter
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
        # TODO: Check how "SNN for time-series analysis" used a linear readout.


    def forward(self, x, hidden_states, prev_obs, readout_states):
        if hidden_states is None:
            raise Exception("hidden_states shouldn't be None when using pytorch environment as a zeroed array is returned in _reset. If using SNNNetwork without rl environment, proceed.")
        
        mem = hidden_states

        if isinstance(mem, torch.Tensor):
            mem = mem.to(x.device)


        if x.dim() == 1:  # x: [Features]
            # Used for data/transition collection
            cur, h, new_prev_obs, new_readout_states =  self.forward_step(x, mem, prev_obs, readout_states)
            return cur, h, new_prev_obs, new_readout_states

        else:  # x: [Sequence, Time, Features]
            # Used when iterating over batches.
            cur, h, new_prev_obs, new_readout_states = self.forward_sequences(x, mem, prev_obs, readout_states)
            return cur, h, new_prev_obs, new_readout_states

    def forward_step(self, x, mem, prev_obs, readout_states):
        """
        Used in the data collection phase where the collector processes individual transitions.
        """
        new_prev_obs = x.clone()
        if self.encoding_in == ENCODING.DELTA:
            x = self.delta_modulator.encode(x)
            # print(f"Forward step: {x}", flush=True)

        cur = self.fc1(x)

        spk = None
        # Loop through the layers up to the output layer.
        for i in range(len(self.hidden_and_output_layers) - 1):
            if i % 2 == 0:  # Spiking layer
                mem_idx = i // 2  # As linear layers don't have a membrane potential - must index these separately.
                # print(f'mem_layer shape: {mem[mem_idx, :].shape}, cur shape: {cur.shape}')
                spk, new_mem = self.hidden_and_output_layers[i](cur, mem[mem_idx, :])
                if self.debug_eval:
                    print(f"Layer {mem_idx}, spikes: {spk.sum()}", flush=True)
                mem[mem_idx, :] = new_mem
            else:  # Linear layer
                cur = self.hidden_and_output_layers[i](spk)

        
        hidden_states = mem 
        # readout_states is the membrane potential of the output layer.
        _, new_readout_states = self.hidden_and_output_layers[-1](cur, readout_states)

        if self.debug_eval:
            print(f"Readout: {new_readout_states}", flush=True)

        return new_readout_states, hidden_states, new_prev_obs, new_readout_states

    def forward_sequences(self, x, hidden_states, prev_obs, readout_states):
        """
        Used by the loss module for a batch of sequences.
        """
        # TODO: This method needs to be fixed in order to work with an arbitrary number of hidden layers.
        # x is a 2D tensor of padded sequences.
        # mem is the hidden states (the membrane potential of the hidden layers)
        
        continuous_x = x
        outputs = []
        new_hidden_states = []
        new_prev_obs = []
        new_readout_states = []

        if hidden_states.dim() == 4:
            mem = hidden_states[:, 0, :, :]  # Only get the first hidden state as the others will be outdated.
        else:
            # This may change if we slice the hidden states before passing them to the loss module.
            raise Exception(f"Expected hidden states to have shape: [NUM_SEQUENCES//NUM_MINI_BATCHES, SEQUENCE_LENGTH, NUM_HIDDEN_LAYERS, NUM_HIDDEN_CELLS], received shape: {hidden_states.shape}")

        if readout_states.dim() == 3:
            readout_states = readout_states[:, 0, :]  # Only take the first readout_state as the subsequent ones will be determined from this initial one.
        else:
            raise Exception(f"Expected hidden states to have shape: [NUM_SEQUENCES//NUM_MINI_BATCHES, SEQUENCE_LENGTH, OUT_FEATURES], received shape: {readout_states.shape}")

        # print(f"Shape of readout: {readout_states.shape}", flush=True)
        
        if self.encoding_in == ENCODING.DELTA:
            # diff = continuous_x - prev_obs
            
            # # Generate the spikes (-1, 0, 1) manually exactly like in forward_step
            # pos_spikes = (diff > self.DELTA_THRESHOLD).float()
            # neg_spikes = (diff < -self.DELTA_THRESHOLD).float() * -1.0
            
            # x = pos_spikes + neg_spikes
            x = self.delta_modulator.encode(continuous_x)
            # print(f"Forward_sequences: {x}", flush=True)


        all_cur = self.fc1(x)  # Linear layer only processes the last dimension of the input tensor (i.e. only the feature dimension is used here).
        # print(f'all_cur_shape: {all_cur.shape}')
        # print(f'x shape: {x.shape}')
        # Loop over all layers up to the output layer
        for t in range(x.size(1)):
            cur = all_cur[:, t, :]  # Outputs of the first linear layer for all sequences (fc1(obs)).

            for i in range(len(self.hidden_and_output_layers) - 1):
                if i % 2 == 0:
                    mem_idx = i // 2
                    spk, new_mem = self.hidden_and_output_layers[i](cur, mem[:, mem_idx, :])
                    # print(f'mem size: {mem.shape}, new_mem: {new_mem.shape}')
                    mem[:, mem_idx, :] = new_mem
                else:
                    cur = self.hidden_and_output_layers[i](spk)

            _, new_readout = self.hidden_and_output_layers[-1](cur, readout_states)
            readout_states = new_readout


            outputs.append(new_readout.clone())
            new_hidden_states.append(mem.clone())
            new_readout_states.append(new_readout.clone())

            # new_hidden_states.append(mem1)

        outputs = torch.stack(outputs, dim = 1)
        new_hidden_states = torch.stack(new_hidden_states, dim=1)
        new_readout_states = torch.stack(new_readout_states, dim=1)

        new_prev_obs = continuous_x
        # new_hidden_states = mem
        # new_hidden_states = torch.stack(new_hidden_states, dim=1)
        return outputs, new_hidden_states, new_prev_obs, new_readout_states

    def load_thresholds(self, path: str, batch_offset: int, agent_idx):
        """Load and update thresholds from file."""
        with open(path, 'r') as f:
            thresholds = json.load(f)
        # str(...) is used when indexing 'thresholds' as JSON only allows string keys which json.dump
        # enforces when the thresholds are saved to file.
        self.delta_modulator.thresholds = torch.tensor(thresholds[str(batch_offset)][str(agent_idx)], device="cuda:0")

    
class AgentEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ("BW", c_uint64),
        ("txRate", c_double),
        ("averageQLength", c_double),
        ("maxQLength", c_uint32),
        ("txRateECN", c_double),
        ("k_min_in", c_double),
        ("k_delta_in", c_double),
        ("p_max_in", c_double)
    ]

class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ("agents", AgentEnv * NUM_AGENTS)
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
        ("agents", AgentAct * NUM_AGENTS)
    ]

class RLEnv(EnvBase):
    def __init__(self, rl_interface, device="cpu"):
        self.batch_size = torch.Size([NUM_AGENTS]) # Number of simultaneous environments running.
        super().__init__(batch_size=self.batch_size, device=device)
        # self.device = device
        self.rl = rl_interface

        self.observation_spec = Composite(
            observation=Unbounded(shape=(*self.batch_size, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),
            hidden_states=Unbounded(
                shape=(*self.batch_size, NUM_HIDDEN_LAYERS, NUM_HIDDEN_CELLS), 
                dtype=torch.float32,
                device=self.device
            ),
            prev_obs=Unbounded(
                shape=(*self.batch_size, NUM_STATE_PARAMS),
                dtype=torch.float32,
                device=self.device
            ),
            readout_states=Unbounded(
                shape=(*self.batch_size, OUT_FEATURES),
                dtype=torch.float32,
                device=self.device
            ),
            shape=self.batch_size,
            device=self.device
        )

        low = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, NUM_ACTIONS)
        high = torch.tensor([500_000.0, 1_000_000.0, 1.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, NUM_ACTIONS)

        self.action_spec = Bounded(
            low=low,
            high=high,  # Currently set max values of K_min and K_max to be the total buffer size.
            shape=(*self.batch_size, NUM_ACTIONS),
            dtype=torch.float32,
            device=self.device
        )
        self.reward_spec = Unbounded(shape=(*self.batch_size, 1), dtype=torch.float32, device=self.device)

        # self.add_truncated_keys()
        self.done_spec = Composite(
            done=Binary(1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            terminated=Binary(1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            truncated=Binary(1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            shape=self.batch_size,
            device=self.device
        )
        self.last_obs = torch.zeros(*self.batch_size, NUM_STATE_PARAMS, dtype=torch.float32, device=self.device)
        self.python2_path = "/home/links/rl624/.conda/envs/hpcc_env/bin/python"
        
        self.pool_id = 1234
        self.rl_id = 2333
        # self.cur_sim_num = 0
        self.ns3_args = {
            "cc": "dcqcn",
            "bw": 100,  # NIC bandwidth
            "rl_ecn_marking": 1,
             # "web_search_320_0.3_100s", # "web_search_5_80_3000s",  # star_5_single_burst_trace  web_search_5_80_100s
            "topo": 'star_5', # "fat", # "star_5",
            "sim_num": 0
        }
        if PERFORM_EVAL:
            self.ns3_args["trace"] = 'web_search_5_80_0.1s'
        else:
            self.ns3_args["trace"] = 'web_search_5_80_3000s'

        self.sim_done = True # Flag to determine whether to start the ns3 simulation.

        self.ns3_process = None

        # Very important scaling to prevent exploding gradients, producing 'nan' actions.
        self.OBS_SCALE = torch.tensor([
            1e11/8.0,   # BW
            1e11,   # txRate
            1e8,    # averageQLength
            1e8,    # maxQLength
            1e11,   # txRateECN
            # N.B. This is normalised based on the buffer size - set to 32MB in NS3.
            high[0][0],   # k_min  - N.B. all agents have the same high/upper bound values.
            high[0][1],   # k_max
            high[0][2]     # p_max
        ], dtype=torch.float32, device=self.device)

        # ============ FOR Debugging ==============
        self.w = 0.5
        self.alpha = 40
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
    
    def set_mode(self, mode):
        print("in set_mode")
        if mode.upper() == "EVAL":
            print("SETTING TRACE")
            self.ns3_args["trace"] = "web_search_5_80_0.1s"
        else:
            pass
    
    def save_reward_history(self):
        with open(f"../reward_data_agent.txt", "w") as f:
            json.dump(self.reward_history, f)


    def start_ns3_simulation(self):
        """
        Start a new ns3 simulation. The previous simulation will be stopped before the new one is started.
        """
        if self.ns3_process and self.ns3_process.poll() is None:
                try:
                    # TODO: document code
                    parent = psutil.Process(self.ns3_process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        # print(f"Killing child: {child.pid} {child.name()}")
                        child.kill()
                    parent.kill()
                    psutil.wait_procs(children)
                    self.ns3_process.wait()
                except psutil.NoSuchProcess:
                    pass
                
                os.system("pkill -9 -f third_in_sync 2>/dev/null")
                time.sleep(1)
                # os.system("pkill -9 -f third_in_sync")
                # os.system("pkill -9 -f waf")
                # os.system("pkill -9 -f run.py")

        print("Free Memory", flush=True)
        py_interface.FreeMemory()

        os.system("ipcs -m | awk '/rl624/ {print $2}' | xargs -r ipcrm -m 2>/dev/null")


        print("Init", flush=True)
        # print(f"isFinish before Init: {self.rl.isFinish()}", flush=True)
        py_interface.Init(self.pool_id + self.ns3_args['sim_num'], 131072)

        print("INITIALISED", flush=True)
        self.rl = py_interface.Ns3AIRL(self.rl_id + self.ns3_args['sim_num'], Env, Act)

        print("Ns3AIRL", flush=True)
        env = os.environ.copy()
        # TODO: Make this cleaner and add logging, so it will be easier to debug.
        env["PATH"] = f"{os.path.dirname(self.python2_path)}:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

        for key in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "PYTHON"]:
            env.pop(key, None)

        args_str = [f'--{key}={value}' for key, value in self.ns3_args.items()]
        if PERFORM_EVAL:
            args_str.append("--perform_eval")

        self.ns3_process = subprocess.Popen([self.python2_path, "run.py", *args_str], env=env, text=True) #, preexec_fn=os.setsid) # stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        self.ns3_args['sim_num'] += 1

    
    def get_obs(self, data):
        new_obs = torch.tensor([
            [
                agent.BW,            
                agent.txRate, 
                agent.averageQLength,
                agent.maxQLength,
                agent.txRateECN,
                agent.k_min_in,
                agent.k_delta_in,
                agent.p_max_in
            ]
            for agent in data.env.agents
        ], device=self.device, dtype=torch.float32)

        new_obs = new_obs.view(*self.batch_size, NUM_STATE_PARAMS)
        # print(new_obs / self.OBS_SCALE)
        return new_obs / self.OBS_SCALE


    def _step(self, td): # td = tensordict
        # print(f"TENSORDICT SHAPE: {td.shape}")
        # action = td["action"]
        # TODO: Need to use the rl object to get the environment info from C++ here and then return control.
        # This is where the communication with the ns3 simulation occurs.
        actions = td["action"] 

        # Send action to NS3 simulation.
        # print("SENDING ACTION")

        with self.rl as data:
            if data is None:
                print("TRIGGERED", flush=True)
                # done = torch.ones(*self.batch_size, dtype=torch.bool, device=self.device)
                # new_obs = self.last_obs
                # reward = torch.zeros(*self.batch_size, 1, dtype=torch.float32, device=self.device)
                
                self.sim_done = True
                return TensorDict({
                    "observation": self.last_obs,
                    "reward": torch.zeros(*self.batch_size, 1, dtype=torch.float32, device=self.device),
                    "done": torch.ones(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                    "terminated": torch.ones(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                    "truncated": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
                }, batch_size=self.batch_size, device=self.device)
            else:
                # =========== DEBUGGING start_ns3_simulation =============
                # print("SETTING done flag", flush=True)
                # self.sim_done = True
                # return TensorDict({
                #     "observation": self.last_obs,
                #     "reward": torch.zeros(*self.batch_size, 1, dtype=torch.float32, device=self.device),
                #     "done": torch.ones(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                #     "terminated": torch.ones(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                #     "truncated": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
                # }, batch_size=self.batch_size, device=self.device)
                # =========================================================
                
                for i in range(NUM_AGENTS):
                    if i == 1:
                        # print(f'Python: k_min={actions[i, 0].item()}, k_max={actions[i, 0].item() + actions[i, 1].item()}, p_max={actions[i, 2].item()}', flush=True)
                        k_min = actions[i, 0].item()
                        if k_min == float('nan'):
                            raise Exception("Threshold is nan")
                    data.act.agents[i].k_min_out = actions[i, 0].item()
                    data.act.agents[i].k_delta_out = actions[i, 1].item()
                    data.act.agents[i].p_max_out = actions[i, 2].item()

                new_obs = self.get_obs(data)
                # print(f"New obs (in step) shape: {new_obs.shape}")
                reward = self._calculate_reward(data.env).view(*self.batch_size, 1)

                self.last_obs = new_obs

            truncated = torch.tensor([[self.rl.isFinish()] for _ in range(NUM_AGENTS)], dtype=torch.bool, device=self.device)
            terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
            done = truncated | terminated 

            # print(new_obs)

            return TensorDict({
                "observation": new_obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            }, batch_size=self.batch_size, device=self.device)

    def _reset(self, tensordict: TensorDictBase | None = None):
        
        if self.sim_done:
            # TODO: probably need a way to kill the old process so that two simulations don't start running causes a problem with the shared memory.
            print("CALLING start_ns3_simulation", flush=True)
            self.start_ns3_simulation()

            print("STARTED ns3 simulation", flush=True)
            
            # TODO: Research how do shared memory locks work properly.
            new_obs = None
            count =0 # FOR DEBUGGING
            while new_obs is None:
                if self.ns3_process.poll() is not None:
                    raise RuntimeError(f"ns-3 process died unexpectedly...")
                if count % 100 == 0:
                    print(f"STUCK in received first obs: count={count}, isFinish: {self.rl.isFinish()}", flush=True)
                with self.rl as data:  # Calls rl.Acquire() and rl.ReleaseMemory() on __enter__ and __exit__ respectively.
                    if data is not None:
                        new_obs = self.get_obs(data)

                        # for i in range(NUM_AGENTS):
                        #     data.act.agents[i].k_min_out = 1500
                        #     data.act.agents[i].k_delta_out = 1500
                        #     data.act.agents[i].p_max_out = 0.1

                        print("RECEIVED FIRST obs", flush=True)
                    else:
                        print(f"Data is None! isFinish={self.rl.isFinish()}", flush=True)
                count += 1
                time.sleep(0.01)

            print("Control back to python", flush=True)
            self.sim_done = False
        else:
            # new_obs = torch.zeros(*self.batch_size, NUM_STATE_PARAMS, dtype=torch.float32, device=self.device)
            # with self.rl as data:
            #     if data is not None:
            #         new_obs = self.get_obs(data)
            #         print(f"New obs shape: {new_obs.shape}")
            new_obs = self.last_obs
        
       
        res = TensorDict(
            {
                "observation": new_obs,
                "hidden_states": torch.zeros((*self.batch_size, NUM_HIDDEN_LAYERS, NUM_HIDDEN_CELLS), dtype=torch.float32, device=self.device),
                "prev_obs": torch.zeros((*self.batch_size, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),  # N.B. this is only used for delta modulation (temporal encoding) input.
                "readout_states": torch.zeros((*self.batch_size, OUT_FEATURES), dtype=torch.float32, device=self.device)
            },
            batch_size=self.batch_size,
            device=self.device
        )

        return res

    def _set_seed(self, seed: int | None = None, static_seed: bool = False):
        torch.manual_seed(seed)
        return seed + 1

    def _calculate_reward(self, env_data):
        # reward = -avgQLength
        rewards = []
        for i in range(NUM_AGENTS):
            txRate = env_data.agents[i].txRate
            avg_q_len = env_data.agents[i].averageQLength
            # Calculate min n, such that E(n) > L, E(n) = a * (2**n)
            # n_power = 0
            # for n in range(0, 10):
            #     n_power = n
            #     # N.B. divide by 1000 as alpha is set in the ACC paper based on the queue length in KB but avg_q_len is in bytes here.
            #     if self.alpha * 2**n > (avg_q_len/1000):
            #         break
            # D = 1 - n_power / 10

            max_qlen = env_data.agents[i].maxQLength
            max_tolerated_max_qlen = 200_000.0
            
            Q_max = max(0.0, 1.0 - (max_qlen / max_tolerated_max_qlen))
            
            max_tolerated_q = 250_000.0
            Q_avg = max(0.0, 1.0 - (avg_q_len / max_tolerated_q))

            T = txRate / 100_000_000_000
            # print(f'txRate: {txRate}, avg_q_len: {avg_q_len}, weighted txRate: {self.w * txRate}, weighted avg_q_len: {(1-self.w) * avg_q_len}')
            # print(f'T: {T}, D: {D}, weighted T: {self.w * T}, weighted D: {(1-self.w) * D}')
            # r = self.w * T + (1-self.w) * D

            # r = -avg_q_len
            w = 0.7
            r = w * T + w * Q_max + (1 - 2 * w) * Q_avg

            # if txRate < 10:
            #     r -= 1000000000
            rewards.append(r)

            # ====== FOR DEBUGGING/ MONITORING =========
            self.reward_history[i]["rewards"].append(r)
            self.reward_history[i]["q_lengths"].append(avg_q_len)
            self.reward_history[i]["throughput"].append(txRate)
            # self.reward_history[i]["n"].append(n_power)
            self.reward_history[i]["T"].append(T)
            # self.reward_history[i]["D"].append(D)
            # TODO: Is this for the previous timestep?
            self.reward_history[i]["k_min"].append(env_data.agents[i].k_min_in)
            self.reward_history[i]["k_max"].append(env_data.agents[i].k_delta_in)
            self.reward_history[i]["p_max"].append(env_data.agents[i].p_max_in)

            # if i == 0:
            #     print(f"Agent {i}: txRate={txRate}, avg_q_len={avg_q_len}", flush=True)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        return rewards

class MultiAgentPolicyWrapper(nn.Module):
    def __init__(self, local_policies):
        super().__init__()
        self.policies = nn.ModuleList(local_policies)

    def forward(self, tensordict):
        output_tds = []
        
        for i, policy in enumerate(self.policies):
            # print(f"Tensordict shape: {tensordict.shape}")
            # print(f"Calling policy: data shape: {tensordict[i].shape}")
            # print(f"Observation shape: {tensordict[i]['observation'].shape}")
            # print(f"Hidden states shape: {tensordict[i]['hidden_states'].shape}")
            
            out_td = policy(tensordict[i])
            output_tds.append(out_td)
            
        batched_output = torch.stack(output_tds, dim=0)
        
        tensordict.update(batched_output)
        
        return tensordict
    

# py_interface.Init(1234, 4096)
# rl = py_interface.Ns3AIRL(2333, Env, Act)

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
env = RLEnv(None, device=device)

saved_models_suffix = f"{'snn' if USE_SNN else 'ann'}_{NUM_HIDDEN_LAYERS}_hidden_layers_{NUM_HIDDEN_CELLS}_hidden_cells_{INPUT_ENCODING.name}_{OUTPUT_ENCODING.name}"

def make_policy_module(agent_idx):
    actor_in_keys = ["observation"]
    if USE_SNN:
        print("Using SNN", flush=True)
        actor_in_keys = ["observation", "hidden_states", "prev_obs", "readout_states"]
        actor_out_keys = ["raw_output", ("next", "hidden_states"), ("next", "prev_obs"), ("next", "readout_states")]
        actor_net = SNNNetwork(NUM_HIDDEN_LAYERS, NUM_HIDDEN_CELLS, 2 * NUM_ACTIONS, INPUT_ENCODING, OUTPUT_ENCODING, debug_eval=False)                
    else:
        actor_in_keys = ["observation"]
        actor_out_keys = ["raw_output"]

        hidden_layers = []
        for _ in range(NUM_HIDDEN_LAYERS):
            hidden_layers.append(nn.Linear(NUM_HIDDEN_CELLS, NUM_HIDDEN_CELLS))
            hidden_layers.append(nn.ReLU())

        actor_net = nn.Sequential(
            nn.Linear(NUM_STATE_PARAMS, NUM_HIDDEN_CELLS),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(NUM_HIDDEN_CELLS, 2 * NUM_ACTIONS)
        )

    # summary(actor_net, input_size=(NUM_STATE_PARAMS,))

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

    return policy_module

def make_value_module():
    value_net = nn.Sequential(
        nn.Linear(NUM_STATE_PARAMS, NUM_HIDDEN_CELLS),
        nn.Tanh(),
        nn.Linear(NUM_HIDDEN_CELLS, NUM_HIDDEN_CELLS),
        nn.Tanh(),
        nn.Linear(NUM_HIDDEN_CELLS, 1)
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return value_module


def get_snn_net(policy):
    '''
    Utility function to make it easier if the structure of the ProbabilisticActor modules change.
    '''
    return policy.module[0].module[0]

local_policies = [ProbabilisticActor(
    module=make_policy_module(i),
    spec=env.action_spec[0],
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={  # Ensure the distributions are between the low and high
        "low": env.action_spec[0].space.low, 
        "high": env.action_spec[0].space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
) for i in range(NUM_AGENTS)]

# If training was interrupted use this to load the previous models and continue the training.
if BATCH_OFFSET != 0 or PERFORM_EVAL:
    for agent_num in range(NUM_AGENTS):
        print(f"Loading policy: {agent_num}", flush=True)
        loaded_state_dict = torch.load(f'../saved_models_{saved_models_suffix}/agent_{agent_num}_batch_{BATCH_OFFSET}_policy.pth')
        local_policies[agent_num].load_state_dict(loaded_state_dict)

        if USE_SNN and INPUT_ENCODING == ENCODING.DELTA:
            print(f'Loading agent {agent_num} thresholds', flush=True)
            snn_net = get_snn_net(local_policies[agent_num])
            snn_net.load_thresholds(f'../saved_models_{saved_models_suffix}/delta_thresholds.json', BATCH_OFFSET, agent_num)

multi_agent_policy_module = MultiAgentPolicyWrapper(local_policies) # So each agent's policy can be executed in the same environment step.

value_modules = [make_value_module() for _ in range(NUM_AGENTS)]

multi_agent_policy_module = multi_agent_policy_module.to(device)
multi_agent_policy_module(env.reset())

for i in range(NUM_AGENTS):
    value_modules[i] = value_modules[i].to(device)
    value_modules[i](env.reset())

# frames_per_batch = 1024

collector = SyncDataCollector(
    env,
    multi_agent_policy_module,
    frames_per_batch=NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH * NUM_AGENTS,
    total_frames=TOTAL_FRAMES,
    split_trajs=False,
    device=device,
    # postproc=extract_collector_signals
)

lr = 3e-4
max_grad_norm = 1.0
gamma = 0.99
lmbda = 0.95

advantage_modules = [GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_modules[i], average_gae=True, device=device,
) for i in range(NUM_AGENTS)]


samplers = [SliceSampler(
    slice_len=SEQUENCE_LENGTH, # Sequence length of 16
    strict_length=True,
    cache_values=True,
    end_key=("next", "done"),
    traj_key=("collector", "traj_ids"),  # Differentiate between samples with the trajectory id.
) for _ in range(NUM_AGENTS)]

replay_buffers = [TensorDictReplayBuffer(
    storage=LazyMemmapStorage(max_size=10000),#LazyTensorStorage(max_size=frames_per_batch),
    sampler=samplers[i],#SamplerWithoutReplacement(),
    batch_size=(NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH) // NUM_MINI_BATCHES, # i.e. num_frames_per_batch / num_mini_batches = num_frames_per_mini_batch
) for i in range(NUM_AGENTS)]

entropy_eps = 0.01 # 1e-4
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)

loss_critic_coeff = 1.0
loss_entropy_coeff = entropy_eps

loss_modules = [ClipPPOLoss(
    actor_network=local_policies[i],
    critic_network=value_modules[i],  # TODO: Does this need to be for individual agents (similar to local_policies[i] above)
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coeff=1.0,
    loss_critic_type="smooth_l1"
) for i in range(NUM_AGENTS)]

optims = [torch.optim.Adam(loss_modules[i].parameters(), lr) for i in range(NUM_AGENTS)]

BATCH_LIMIT = 40  # TODO: remove. This is temporary whilst the "reset sim" functionality does not work correctly - to end the training before the reset occurs (found empirically)

total_training_frames = BATCH_LIMIT * NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
    optims[i], total_training_frames // (NUM_DESIRED_SEQUENCES * SEQUENCE_LENGTH), 0.0
) for i in range(NUM_AGENTS)]

logs = defaultdict(list)
pbar = tqdm(total=TOTAL_FRAMES)
eval_str = ""

delta_modulation_thresholds = {}

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

        cur_loss_values = {}
        for agent_num in range(NUM_AGENTS):
            replay_buffers[agent_num].extend(tensordict_data[agent_num])
            advantage_modules[agent_num](tensordict_data[agent_num])
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            update_steps = 0

            for epoch in range(NUM_EPOCHS):
                # print(f"Agent {agent_num}", flush=True)
                print(f"Agent: {agent_num}, Batch {i + BATCH_OFFSET}, epoch {epoch}", flush=True)
                for _ in range(NUM_DESIRED_SEQUENCES // NUM_MINI_BATCHES): 
                    subdata = replay_buffers[agent_num].sample()
                    # print(f"batch_size: {tensordict_data[agent_num].shape}, subdata shape: {subdata.shape}, num_sequences: {NUM_DESIRED_SEQUENCES}, sl: {SEQUENCE_LENGTH}")
                    subdata = subdata.view(NUM_DESIRED_SEQUENCES // NUM_MINI_BATCHES, SEQUENCE_LENGTH)

                    # print(f"Subdata shape: {subdata.shape}")

                    loss_vals = loss_modules[agent_num](subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    total_policy_loss += loss_vals["loss_objective"]
                    total_value_loss += loss_vals["loss_critic"]
                    total_entropy_loss += loss_vals["loss_entropy"]

                    update_steps += 1

                    # ADD THESE CHECKS
                    if torch.isnan(loss_value):
                        print(f"NaN loss at batch {batch_num}, epoch {epoch}, agent {agent_num}", flush=True)
                        print(f"  loss_objective: {loss_vals['loss_objective']}", flush=True)
                        print(f"  loss_critic: {loss_vals['loss_critic']}", flush=True)
                        print(f"  loss_entropy: {loss_vals['loss_entropy']}", flush=True)
                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()

                    # grad_norm = torch.nn.utils.clip_grad_norm_(loss_modules[agent_num].parameters(), max_grad_norm)
                    # if torch.isnan(grad_norm):
                    #     print(f"NaN grad norm at batch {batch_num}, epoch {epoch}, agent {agent_num}", flush=True)

                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_modules[agent_num].parameters(), max_grad_norm)
                    optims[agent_num].step()
                    optims[agent_num].zero_grad()

                    # Check weights after update
                    # for name, param in local_policies[agent_num].named_parameters():
                    #     if torch.isnan(param).any():
                    #         print(f"NaN weight in {name} at batch {batch_num}, epoch {epoch}, agent {agent_num}", flush=True)
                    #         break
            # log losses here.
            avg_policy_loss = total_policy_loss / update_steps
            avg_value_loss = total_value_loss / update_steps
            avg_entropy_loss = total_entropy_loss / update_steps
            cur_loss_values[agent_num] = [avg_policy_loss.item(), avg_value_loss.item(), avg_entropy_loss.item()]

            replay_buffers[agent_num].empty()

        loss_values.append(cur_loss_values)

        schedulers[agent_num].step()
            # env.save_reward_history()
           

        if batch_num % 10 == 0:
            fig, ax = plt.subplots()

            policy_loss = [loss_values[i][agent_num][0] for i in range(len(loss_values))]
            value_loss = [loss_values[i][agent_num][1] for i in range(len(loss_values))]
            entropy_loss = [loss_values[i][agent_num][2] for i in range(len(loss_values))]

            fig, ax = plt.subplots()
            ax.plot(policy_loss)
            fig.savefig(f'../losses/policy_loss_{saved_models_suffix}.png')

            fig, ax = plt.subplots()
            ax.plot(value_loss)
            fig.savefig(f'../losses/value_loss_{saved_models_suffix}.png')

            fig, ax = plt.subplots()
            ax.plot(entropy_loss)
            fig.savefig(f'../losses/entropy_loss_{saved_models_suffix}.png')

            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                print("RUNNING EVAL ROLLOUT", flush=True)
                eval_rollout = env.rollout(500, multi_agent_policy_module)
                # ========== DEBUGGING =============
                # Check for nan in actions and rewards
                actions = eval_rollout["action"]
                rewards = eval_rollout["next", "reward"]
                print(f"NaN in actions: {torch.isnan(actions).any()}", flush=True)
                print(f"NaN in rewards: {torch.isnan(rewards).any()}", flush=True)
                
                # Find the first step where nan appears
                nan_steps = torch.isnan(actions).any(dim=-1).any(dim=0)  # shape: [steps]
                if nan_steps.any():
                    first_nan_step = nan_steps.nonzero()[0].item()
                    print(f"First NaN action at step: {first_nan_step}", flush=True)
                    print(f"Action at step {first_nan_step}: {actions[:, first_nan_step]}", flush=True)
                    print(f"Observation at step {first_nan_step}: {eval_rollout['observation'][:, first_nan_step]}", flush=True)
                # ===================================

                print(eval_rollout, flush=True)
                print(eval_rollout["next", "reward"], flush=True)
                mean_reward = eval_rollout["next", "reward"].mean(dim=1)
                q_lengths = eval_rollout["observation"][:, :, OBS.averageQLength.value]
                mean_q_length = q_lengths.mean(dim=1)
                print("MEAN REWARD: ", mean_reward, flush=True)
                print("MEAN Q LENGTH: ", mean_q_length, flush=True)
                for agent_num in range(NUM_AGENTS):
                    tracked_data[agent_num][i] = {
                        "mean_reward": mean_reward[agent_num].item(),
                        "mean_q_length": mean_q_length[agent_num].item()
                    }
                print("TRACKED DATA: ", tracked_data, flush=True)

                with open('../trained_models_data.txt', 'w') as f:
                    json.dump(tracked_data, f)

                    # for agent_num in range(NUM_AGENTS):
                    #     f.write(f"Agent {agent_num}:\n")
                    #     for batch_num, data in tracked_data[agent_num].items():
                    #         f.write(f"{batch_num},{data['mean_reward']},{data['mean_q_length']}\n")
                
                for agent_num in range(NUM_AGENTS):
                    torch.save(local_policies[agent_num].state_dict(), f'../saved_models_{saved_models_suffix}/agent_{agent_num}_batch_{i}_policy.pth')
                
                if USE_SNN and INPUT_ENCODING == ENCODING.DELTA:
                    delta_modulation_thresholds[batch_num] = {
                        agent_idx: get_snn_net(local_policies[agent_idx]).delta_modulator.thresholds.cpu().detach().tolist() for agent_idx in range(NUM_AGENTS)
                    }
                    
                    with open(f'../saved_models_{saved_models_suffix}/delta_thresholds.json', 'w') as f:
                        json.dump(delta_modulation_thresholds, f)

    with open('../../trained_models_data.txt', 'w') as f:
        for agent_num in range(NUM_AGENTS):
            f.write(f"Agent {agent_num}:\n")
            for batch_num, data in tracked_data[agent_num].items():
                f.write(f"{batch_num},{data['mean_reward']},{data['mean_q_length']}\n")
else:
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        print("RUNNING EVAL ROLLOUT", flush=True)
        # env.start_ns3_simulation()  # This will stop the previous simulation and start a new one.
        eval_rollout = env.rollout(10_000_000, multi_agent_policy_module) # Run until the simulation ends
        print("DONE ROLLOUT", flush=True)
        env.save_reward_history()


            
