
from ctypes import Structure, c_double, c_int, c_uint64
import py_interface
from concurrent.futures import ProcessPoolExecutor
import sys
from tensordict import TensorDict, TensorDictBase
import torch
from torch import nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data import Bounded, Unbounded
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
import numpy as np
import torch.multiprocessing as mp
from torchrl.envs import EnvBase
from torchrl.data import Composite
from torchrl.collectors import SyncDataCollector

# Define shared memory structures
# Env - data coming from ns3 (C++)

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import multiprocessing

class SNNNetwork(nn.Module):
    # TODO: Add a batch dimension so that parallel training is possible.
    def __init__(self, num_hidden, out_features):
        super().__init__()
        # initialize layers
        num_hidden = 64
        BETA = 0.8  # N.B. to make this a learnable parameter, wrap in nn.Parameter
        self.fc1 = nn.LazyLinear(num_hidden)
        self.lif1 = snn.Leaky(beta=BETA)
        self.fc2 = nn.LazyLinear(out_features)

        # TODO: Check how "SNN for time-series analysis" used a linear readout.


    def forward(self, x, hidden_states = None):
        # TODO: generalise mem1 (membrane potential) for all hidden states.
        if hidden_states is None:
            mem1 = self.lif1.reset_mem()
        else:
            mem1 = hidden_states

        if x.dim() == 2:  # x: [Batch, Features]
            cur, h =  self.forward_step(x, mem1)
            return cur, h
        else:  # x: [Batch, Time, Features]
            cur, h = self.forward_sequences(x, hidden_states)
            return cur, h

    def forward_step(self, x, mem1):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)

        hidden_states = mem1  # TODO: ensure this works for different hidden layer configurations.
        return cur2, hidden_states

    def forward_sequences(self, x, hidden_states):
        # x is a 2D tensor of padded sequences.
        # mem1 is the hidden states (the membrane potential of the first and only hidden layer - to be generalised for deeper models)
        # print(f"Sequence shape: {x.shape}")
        # raise Exception("HELP")
        outputs = []
        new_hidden_states = []
        mem1 = hidden_states[:, 0]
        all_cur1 = self.fc1(x)
        for t in range(x.size(1)):
            xt = all_cur1[:, t, :]
            spk1, mem1 = self.lif1(xt, mem1)
            cur2 = self.fc2(spk1)
            outputs.append(cur2)
            new_hidden_states.append(mem1)

        outputs = torch.stack(outputs, dim = 1)
        new_hidden_states = torch.stack(new_hidden_states, dim=1)
        return outputs, new_hidden_states
    

'''
From the ACC paper, the following values are needed for the reward function:
r = w_1 * T(R) + w_2 * D(L)

T(R) = txRate / BW  - denotes the utilization of a link
txRate = average throughput of one egress queue (output data rate)
BW = Bandwidth of link

D(L) = 1 - n/10, n = argmin_n (E(n)>= L), forall n in [0,9]
L = average queue length
D is a stepping function.


Therefore from the environment: BW, txRate, and L are needed.

For the state:
- Queue Length
- output data rate for each link (txRate)
- output rate of ECN markedpackets for each link (txRate(m))
- current ECN setting

Therefore for both the state and reward function, we need the following values from the simulation:
- BW
- txRate - output data rate from an egress port
- Queue Length
- txRate(m) - output rate of ECN marked packets for each link
- current ECN setting
'''

NUM_ACTIONS = 3
NUM_STATE_PARAMS = 7
NUM_HIDDEN_CELLS = 64

TOTAL_BUFFER_SIZE = 32 # MB

from torchrl.data import Binary

class RLEnv(EnvBase):
    def __init__(self, rl_interface, device="cpu"):
        self.batch_size = torch.Size([1]) # Number of simultaneous environments running.
        super().__init__(batch_size=self.batch_size, device=device)
        # self.device = device
        self.rl = rl_interface

        self.observation_spec = Composite(
            observation=Unbounded(shape=(*self.batch_size, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),
            hidden_states=Unbounded(
                shape=(*self.batch_size, NUM_HIDDEN_CELLS), 
                dtype=torch.float32,
                device=self.device
            ),
            shape=self.batch_size,
            device=self.device
        )

        low = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, NUM_ACTIONS)
        high = torch.tensor([TOTAL_BUFFER_SIZE * 1000, TOTAL_BUFFER_SIZE * 1000, 1.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, NUM_ACTIONS)

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

    def _step(self, td): # td = tensordict
        # action = td["action"]
        # TODO: Need to use the rl object to get the environment info from C++ here and then return control.
        # This is where the communication with the ns3 simulation occurs.
        actions = td["action"]
        # N.B. indexing with '...' accesses the last dimension - this makes the code still function when using parallel environments.
        k_min_out, k_gap, p_max_out = actions[...,0].item(), actions[...,1].item(), actions[...,2].item()

        k_max_out = k_min_out + k_gap

        if k_max_out > TOTAL_BUFFER_SIZE * 1000:
            k_max_out = TOTAL_BUFFER_SIZE * 1000

        # Send action to NS3 simulation.
        # print("SENDING ACTION")
        with self.rl as data:
            if data is None:
                done = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            else:
                data.act.k_min_out = k_min_out
                data.act.k_max_out = k_max_out
                data.act.p_max_out = p_max_out

        # print("RETRIEVING OBSERVATION")
        with self.rl as data:
            # new_obs = torch.tensor([data.env.BW, data.env.txRate, data.env.averageQLength, 
            #                         data.env.txRateECN, data.env.k_min_in, data.env.k_max_in, 
            #                         data.env.p_max_in], device=self.device, dtype=torch.float32)
            new_obs = torch.tensor([
                data.env.BW / 1e10,              # Scale by max expected bandwidth
                data.env.txRate / 1e10, 
                data.env.averageQLength / 1000,  # Scale by typical max queue
                data.env.txRateECN / 1e10,
                data.env.k_min_in / 1000,
                data.env.k_max_in / 1000,
                data.env.p_max_in
            ], device=self.device, dtype=torch.float32)
            new_obs = new_obs.view(*self.batch_size, NUM_STATE_PARAMS)
            reward = self._calculate_reward(data.env).view(*self.batch_size, 1)

        truncated = torch.tensor([[self.rl.isFinish()]], dtype=torch.bool, device=self.device)
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
        print("RESETTING ENVIRONMENT...")
        return TensorDict(
            {
                "observation": torch.rand((*self.batch_size, NUM_STATE_PARAMS), dtype=torch.float32, device=self.device),
                "hidden_states": torch.rand((*self.batch_size, NUM_HIDDEN_CELLS), dtype=torch.float32, device=self.device)
            },
            batch_size=self.batch_size,
            device=self.device
        )

    def _set_seed(self, seed: int | None = None, static_seed: bool = False):
        torch.manual_seed(seed)
        return seed + 1

    def _calculate_reward(self, env_data):
        return torch.ones(self.batch_size, dtype=torch.float32, device=self.device)


# Define structures for IPC with C++ simulation.
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('BW', c_uint64),
        ('txRate', c_double),
        ('averageQLength', c_double),
        ('txRateECN', c_double),
        ('k_min_in', c_double),
        ('k_max_in', c_double),
        ('p_max_in', c_double)
    ]
# Act - the action produced from RL agent (Python)
class Act(Structure):
    _pack_ = 1
    _fields_ = [
        # N.B. remember that ECN parameters are a mapping in the simulation.
        ('k_min_out', c_double),
        ('k_max_out', c_double),
        ('p_max_out', c_double)
    ]

mempool_key = 1234  # This is the 'Pool' key
mem_size = 1024 * 1024     # Size in bytes
start_block_key = 2333   
 
frames_per_batch = 1024
total_frames = 10_000
sequence_length = 16

# TODO: Make the 'num_agents' dynamic by getting the value from the NS-3 simulation.
def agent_listener(block_key):
    py_interface.Init(mempool_key, mem_size)
    rl = py_interface.Ns3AIRL(block_key, Env, Act)

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    env = RLEnv(rl, device=device)

    actor_net = SNNNetwork(NUM_HIDDEN_CELLS, 2 * NUM_ACTIONS)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "hidden_states"],
        out_keys=["raw_output", ("next", "hidden_states")] # This will move the output hidden state for timestep t-1  to the hidden_state for timestep t.
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

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={  # Ensure the distributions are between the low and high
            "low": env.action_spec.space.low, 
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = nn.Sequential(
        nn.Linear(NUM_STATE_PARAMS, NUM_HIDDEN_CELLS),
        nn.Tanh(),
        nn.Linear(NUM_HIDDEN_CELLS, NUM_HIDDEN_CELLS),
        nn.Tanh(),
        nn.Linear(NUM_HIDDEN_CELLS, 1)
    )

    policy_module = policy_module.to(device)
    value_net = value_net.to(device)
    
    policy_module(env.reset())

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,

    )

    print(f"Python Agent, block: {block_key}: Listening for ns-3...")
    
    for i, batch in enumerate(collector):
        print("PRINTING KEYS")
        print(batch.keys())

        

# TODO: Get this info from the simulation
NUM_AGENTS = 5  # There will be an agent for every port on every switch

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    block_ids = [start_block_key + i for i in range(NUM_AGENTS)]

    print(f"Launching {NUM_AGENTS} parallel agents...")

    processes = []
    for i in range(NUM_AGENTS):
        block_id = start_block_key + i
        p = mp.Process(target=agent_listener, args=(block_id,))
        p.start()
        processes.append(p)
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
    finally:
        py_interface.FreeMemory()