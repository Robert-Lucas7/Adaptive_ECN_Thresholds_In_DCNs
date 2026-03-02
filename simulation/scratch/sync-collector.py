from collections import defaultdict

from tensordict import TensorDict, TensorDictBase
import torch
import torch.multiprocessing as mp
import py_interface
from ctypes import Structure, c_double, c_uint64
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

class OBS(Enum):
    BW = 0
    txRate = 1
    averageQLength = 2
    txRateECN = 3
    k_min_in = 4
    k_max_in = 5
    p_max_in = 6

NUM_ACTIONS = 3
NUM_STATE_PARAMS = 7
NUM_HIDDEN_CELLS = 64
NUM_AGENTS = 5
FRAMES_PER_BATCH = 1024
TOTAL_FRAMES = 5_000_000
SEQUENCE_LENGTH = 16
TOTAL_BUFFER_SIZE = 1000  # TODO: REMOVE

class SNNNetwork(nn.Module):
    # TODO: Add a batch dimension so that parallel training is possible.
    def __init__(self, num_hidden, out_features):
        super().__init__()
        # initialize layers
        num_hidden = 64
        BETA = 0.8  # N.B. to make this a learnable parameter, wrap in nn.Parameter
        self.fc1 = nn.Linear(NUM_STATE_PARAMS, num_hidden)
        self.lif1 = snn.Leaky(beta=BETA)
        self.fc2 = nn.Linear(num_hidden, out_features)

        # TODO: Check how "SNN for time-series analysis" used a linear readout.


    def forward(self, x, hidden_states = None):
        # TODO: generalise mem1 (membrane potential) for all hidden states.
        if hidden_states is None:
            mem1 = self.lif1.reset_mem()
        else:
            mem1 = hidden_states

        if isinstance(mem1, torch.Tensor):
            mem1 = mem1.to(x.device)


        if x.dim() == 1:  # x: [Features]
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
    
class AgentEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ("BW", c_uint64),
        ("txRate", c_double),
        ("averageQLength", c_double),
        ("txRateECN", c_double),
        ("k_min_in", c_double),
        ("k_max_in", c_double),
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
        ("k_max_out", c_double),
        ("p_max_out", c_double)
    ]

class Act(Structure):
    _pack_ = 1
    _fields_ = [
        ("agents", AgentAct * NUM_AGENTS)
    ]

class RLEnv(EnvBase):
    def __init__(self, rl_interface, device="cpu"):
        self.batch_size = torch.Size([5]) # Number of simultaneous environments running.
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
        high = torch.tensor([3000.0, 3000.0, 1.0], dtype=torch.float32, device=self.device).expand(*self.batch_size, NUM_ACTIONS)

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
        self.ns3_args = {
            "cc": "dcqcn",
            "bw": 100,
            "rl_ecn_marking": 1,
            "trace": "web_search_5_80_2s",  # star_5_single_burst_trace  web_search_5_80_100s
            "topo": "star_5"
        }
        self.sim_done = True # Flag to determine whether to start the ns3 simulation.
    
    def start_ns3_simulation(self):
        env = os.environ.copy()
        # TODO: Make this cleaner and add logging, so it will be easier to debug.
        env["PATH"] = f"{os.path.dirname(self.python2_path)}:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

        for key in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "PYTHON"]:
            env.pop(key, None)

        args_str = [f'--{key}={value}' for key, value in self.ns3_args.items()]
        subprocess.Popen([self.python2_path, "run.py", *args_str], env=env, text=True)
    
    def get_obs(self, data):
        new_obs = torch.tensor([
            [
                agent.BW / 1e10,              # Scale by max expected bandwidth
                agent.txRate / 1e10, 
                agent.averageQLength / 1000,  # Scale by typical max queue
                agent.txRateECN / 1e10,
                agent.k_min_in / 1000,
                agent.k_max_in / 1000,
                agent.p_max_in
            ]
            for agent in data.env.agents
        ], device=self.device, dtype=torch.float32)

        new_obs = new_obs.view(*self.batch_size, NUM_STATE_PARAMS)
        return new_obs


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
                done = torch.ones(*self.batch_size, dtype=torch.bool, device=self.device)
                new_obs = self.last_obs
                reward = torch.zeros(*self.batch_size, 1, dtype=torch.float32, device=self.device)
                
                self.sim_done = True
            else:
                for i in range(NUM_AGENTS):
                    data.act.agents[i].k_min_out = actions[i, 0].item()
                    data.act.agents[i].k_max_out = actions[i, 1].item()
                    data.act.agents[i].p_max_out = actions[i, 2].item()

                new_obs = self.get_obs(data)
                # print(f"New obs (in step) shape: {new_obs.shape}")
                reward = self._calculate_reward(data.env).view(*self.batch_size, 1)

                self.last_obs = new_obs

            truncated = torch.tensor([[self.rl.isFinish()] for _ in range(NUM_AGENTS)], dtype=torch.bool, device=self.device)
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
            self.start_ns3_simulation()

            print("STARTED ns3 simulation")
            
            # TODO: Research how to shared memory locks work properly.
            new_obs = None
            while new_obs is None:
                print("GETTING FIRST obs")
                with self.rl as data:  # Calls rl.Acquire() and rl.ReleaseMemory() on __enter__ and __exit__ respectively.
                    if data is not None:
                        new_obs = self.get_obs(data)
                        print("RECEIVED FIRST obs")

            print("Control back to python")
            self.sim_done = False
        else:
            new_obs = torch.zeros(*self.batch_size, NUM_STATE_PARAMS, dtype=torch.float32, device=self.device)
            with self.rl as data:
                if data is not None:
                    new_obs = self.get_obs(data)
                    print(f"New obs shape: {new_obs.shape}")
        
       
        res = TensorDict(
            {
                "observation": new_obs,
                "hidden_states": torch.zeros((*self.batch_size, NUM_HIDDEN_CELLS), dtype=torch.float32, device=self.device)
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
        rewards = torch.tensor([-env_data.agents[i].averageQLength for i in range(NUM_AGENTS)], dtype=torch.float32, device=self.device)
        return rewards

class MultiAgentPolicyWrapper(nn.Module):
    def __init__(self, local_policies):
        super().__init__()
        self.policies = nn.ModuleList(local_policies)

    def forward(self, tensordict):
        output_tds = []
        # print(tensordict.shape)
        for i, policy in enumerate(self.policies):
            out_td = policy(tensordict[i])
            output_tds.append(out_td)
            
        batched_output = torch.stack(output_tds, dim=0)
        
        tensordict.update(batched_output)
        
        return tensordict
    

py_interface.Init(1234, 4096)
rl = py_interface.Ns3AIRL(2333, Env, Act)

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

local_policies = [ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec[0],
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={  # Ensure the distributions are between the low and high
        "low": env.action_spec[0].space.low, 
        "high": env.action_spec[0].space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
) for _ in range(NUM_AGENTS)]

multi_agent_policy_module = MultiAgentPolicyWrapper(local_policies) # So each agent's policy can be executed in the same environment step.


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

multi_agent_policy_module = multi_agent_policy_module.to(device)
value_module = value_module.to(device)

multi_agent_policy_module(env.reset())
value_module(env.reset())

frames_per_batch = 1024
total_frames = TOTAL_FRAMES

collector = SyncDataCollector(
    env,
    multi_agent_policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    # postproc=extract_collector_signals
)

lr = 3e-4
max_grad_norm = 1.0
gamma = 0.99
lmbda = 0.95

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
)


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
    batch_size=frames_per_batch // SEQUENCE_LENGTH,  # 10 sequences per batch?
) for i in range(NUM_AGENTS)]

entropy_eps = 1e-4
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)

loss_critic_coeff = 1.0
loss_entropy_coeff = entropy_eps

loss_modules = [ClipPPOLoss(
    actor_network=local_policies[i],
    critic_network=value_module,  # TODO: Does this need to be for individual agents (similar to local_policies[i] above)
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coeff=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coeff=1.0,
    loss_critic_type="smooth_l1"
) for i in range(NUM_AGENTS)]

optims = [torch.optim.Adam(loss_modules[i].parameters(), lr) for i in range(NUM_AGENTS)]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
    optims[i], total_frames // frames_per_batch, 0.0
) for i in range(NUM_AGENTS)]

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

perform_eval = True

batch_offset = 20  # If training was interrupted use this to load the previous models and continue the training.
if batch_offset != 0:
    for agent_num in range(NUM_AGENTS):
        print(f"Loading policy: {agent_num}")
        loaded_state_dict = torch.load(f'../saved_models/agent_{agent_num}_batch_{batch_offset}_policy.pth')
        local_policies[agent_num].load_state_dict(loaded_state_dict)

# DO TRAINING!
if not perform_eval:
    NUM_EPOCHS = 10
    tracked_data = {agent_num: {} for agent_num in range(NUM_AGENTS)}
    print("REACHED LOOP")

    for i, tensordict_data in enumerate(collector):
        print("Batch collected")
        print(tensordict_data)
        batch_num = i + batch_offset  # TEMPORARY: interrupted training
        for epoch in range(NUM_EPOCHS):
            advantage_module(tensordict_data)
            print(f"Batch {i + batch_offset}, epoch {epoch}")
            for agent_num in range(NUM_AGENTS):
                print(f"Agent {agent_num}")
                replay_buffers[agent_num].extend(tensordict_data[agent_num])
                
                for _ in range(frames_per_batch // SEQUENCE_LENGTH):
                    subdata = replay_buffers[agent_num].sample()
                    subdata = subdata.view(4, SEQUENCE_LENGTH)

                    loss_vals = loss_modules[agent_num](subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_modules[agent_num].parameters(), max_grad_norm)
                    optims[agent_num].step()
                    optims[agent_num].zero_grad()
                schedulers[agent_num].step()
                
        if batch_num % 5 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                print("RUNNING EVAL ROLLOUT")
                eval_rollout = env.rollout(1000, multi_agent_policy_module)
                mean_reward = eval_rollout["next", "reward"].mean(dim=1)
                q_lengths = eval_rollout["observation"][:, :, OBS.averageQLength.value]
                mean_q_length = q_lengths.mean(dim=1)
                print("MEAN REWARD: ", mean_reward)
                print("MEAN Q LENGTH: ", mean_q_length)
                for agent_num in range(NUM_AGENTS):
                    tracked_data[agent_num][i] = {
                        "mean_reward": mean_reward[agent_num].item(),
                        "mean_q_length": mean_q_length[agent_num].item()
                    }
                print("TRACKED DATA: ", tracked_data)

                with open('../trained_models_data.txt', 'a') as f:
                    for agent_num in range(NUM_AGENTS):
                        f.write(f"Agent {agent_num}:\n")
                        for batch_num, data in tracked_data[agent_num].items():
                            f.write(f"{batch_num},{data['mean_reward']},{data['mean_q_length']}\n")
                
                for agent_num in range(NUM_AGENTS):
                    torch.save(local_policies[agent_num].state_dict(), f'../saved_models/agent_{agent_num}_batch_{i}_policy.pth')

    with open('../../trained_models_data.txt', 'w') as f:
        for agent_num in range(NUM_AGENTS):
            f.write(f"Agent {agent_num}:\n")
            for batch_num, data in tracked_data[agent_num].items():
                f.write(f"{batch_num},{data['mean_reward']},{data['mean_q_length']}\n")
else:
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        print("RUNNING EVAL ROLLOUT")
        eval_rollout = env.rollout(1000, multi_agent_policy_module)


            
