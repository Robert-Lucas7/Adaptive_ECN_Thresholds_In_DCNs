from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import py_interface
from ctypes import Structure, c_double, c_uint64
import multiprocessing
from torchrl.data import Bounded
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
FRAMES_PER_BATCH = 1024
TOTAL_FRAMES = 100_000
SEQUENCE_LENGTH = 16


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

        if x.dim() == 1:  # x: [Batch, Features]
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

# Define structures for IPC with C++ simulation.
class Env(Structure):
    _pack_ = 1
    _fields_ = [
        ('agentNum', c_uint64),
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

def update_model_weights_from_shared_memory(policy, value, models_td, agent_num, device):
    policy_params = models_td["policy_params"][agent_num].to(device)

    nested_policy_params = policy_params.unflatten_keys(".")
    nested_policy_params.to_module(policy)

    value_params = models_td["value_params"][agent_num].to(device)

    nested_value_params = value_params.unflatten_keys(".")

    nested_value_params.to_module(value)

    policy_buffers = models_td["policy_buffers"][agent_num]
    value_buffers = models_td["value_buffers"][agent_num]

    policy.load_state_dict(policy_buffers.to(device).to_dict(), strict=False)
    value.load_state_dict(value_buffers.to(device).to_dict(), strict=False)

    return nested_policy_params, nested_value_params


def do_work(agent_num, experience_td, models_td, collection_barrier, training_barrier, save_trained_params_barrier):
    # An individual RL agent - used for training.
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    local_policy_net = create_policy_module(device)
    local_value_net = create_value_module()

    nested_policy_params, nested_value_params = update_model_weights_from_shared_memory(local_policy_net, local_value_net, models_td, agent_num, device)

    params_to_optimize = [
        p.detach().requires_grad_(True) 
        for p in 
        list(nested_policy_params.values(True, True)) + list(nested_value_params.values(True, True))
    ]
    
    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=3e-4
    )

    loss_module = ClipPPOLoss(actor=local_policy_net, critic=local_value_net, device=device)

    gamma = 0.99
    lmbda = 0.95

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=local_value_net, average_gae=True, device=device,
    )
    batch_num = 0
    total_frames = 0
    while True:
        print(f'{agent_num}: Training process started...')
        collection_barrier.wait()

        batch_data = experience_td[:, agent_num].clone().to(device)
        
        advantage_module(batch_data)
        sequence_data = batch_data.view(FRAMES_PER_BATCH // SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        
        for epoch in range(10): # 10 epochs
            print(f'{agent_num}: epoch {epoch}')
            for _ in range(FRAMES_PER_BATCH // SEQUENCE_LENGTH):  # Frames per batch // sequence_length
                loss_vals = loss_module(sequence_data)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

        with torch.no_grad():
            models_td["policy_params"][agent_num].copy_(nested_policy_params.flatten_keys("."))
            models_td["value_params"][agent_num].copy_(nested_value_params.flatten_keys("."))

        total_frames += FRAMES_PER_BATCH
        if batch_num % 5 == 0: #total_frames >= TOTAL_FRAMES:
            print(f"{agent_num}: Finished training...")
            save_trained_params_barrier.wait()
            print('Agent 0 saving model params...')
            if agent_num == 0:
                models_td.save(f'../../saved_models/saved_models_batch_{batch_num}.pt')
            
        batch_num += 1

        if total_frames >= TOTAL_FRAMES:
            print("ALL TRAINING COMPLETED")
            save_trained_params_barrier.wait()
            if agent_num == 0:
                models_td.save(f'../../saved_models/saved_models_batch_finished.pt')
            break

        training_barrier.wait()

def calculate_reward(obs):
    return -obs[OBS.averageQLength.value]


'''
This data collection should ensure the data follows this format (as it is required to calculate the advantage):
tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``

'''
def do_data_collection(experience_td, models_td, batch_size, num_agents, collection_barrier, training_barrier):
    try:
        print("Collecting data...")
        py_interface.Init(1234, 4096)
        rl = py_interface.Ns3AIRL(2333, Env, Act)

        is_fork = multiprocessing.get_start_method() == "fork"
        device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )

        local_policies = [create_policy_module(device) for _ in range(num_agents)]
        local_values = [create_value_module() for _ in range(num_agents)]

        for i in range(num_agents):
            update_model_weights_from_shared_memory(local_policies[i], local_values[i], models_td, i, device)
        
        # Start simulation here - The simulation allocates the memory for the IPC.
        # rl = py_interface.Ns3AIRL(2333, Env, Act)
        agent_obs_num = {agent_num: 0 for agent_num in range(num_agents)}

        agent_obs_data = TensorDict({
            "observation": torch.zeros(num_agents, NUM_STATE_PARAMS),
            "action": torch.zeros(num_agents, NUM_ACTIONS),
            "next_obs": torch.zeros(num_agents, NUM_STATE_PARAMS),
            "is_first_obs": torch.ones(num_agents, dtype=torch.bool),
            "hidden_states": torch.zeros(num_agents, NUM_HIDDEN_CELLS)
        }, batch_size=[num_agents], device=device)
        
        while not (rl.isFinish()):
            with rl as data:
                if data is None:
                    # Simulation has finished for the specified trace file - restart it or generate new trace and run new simulation.
                    print("DATA IS NONE")
                    raise Exception("DATA IS NONE")
                    pass
                
                agent_num = data.env.agentNum
                obs_number = agent_obs_num[agent_num]
                # print(f'Agent num: {agent_num}')
                if agent_obs_num[agent_num] >= batch_size:
                    data.act.k_min_out = data.env.k_min_in
                    data.act.k_max_out = data.env.k_min_in
                    data.act.p_max_out = data.env.p_max_in
                    continue

                # print(td[:, agent_num])
                current_obs = torch.tensor([
                    data.env.BW / 1e10,              # Scale by max expected bandwidth
                    data.env.txRate / 1e10, 
                    data.env.averageQLength / 1000,  # Scale by typical max queue
                    data.env.txRateECN / 1e10,
                    data.env.k_min_in / 1000,
                    data.env.k_max_in / 1000,
                    data.env.p_max_in
                ], device=device, dtype=torch.float32)

                current_agent_data = agent_obs_data[agent_num]
                # For the first iteration after the simulation has been restarted.
                if current_agent_data["is_first_obs"]:
                    current_agent_data["observation"] = current_obs
                    # agent_obs_data[agent_num]["action"] = torch.tensor([400.0, 1600.0, 0.2])  # TODO: get this from the current policy.
                    current_agent_data["is_first_obs"].fill_(False)
                    local_policies[agent_num](current_agent_data)
                else:
                    current_agent_data["next_obs"].copy_(current_obs)
                    # Set the next_obs field and store the combined transistion experience in the experience_td.
                    # Then switch the next and current obs ready for the next iteration.
                    experience_td[obs_number, agent_num]["observation"].copy_(current_agent_data["observation"])

                    local_policies[agent_num](current_agent_data)

                    experience_td[obs_number, agent_num]["action"].copy_(current_agent_data["action"])
                    # TODO: MUST STORE loc, scale, sample_log_prob as keys for the training

                    experience_td[obs_number, agent_num]["next"]["observation"].copy_(current_obs.cpu())
                    experience_td[obs_number, agent_num]["next"]["reward"].copy_(calculate_reward(current_obs))
                    experience_td[obs_number, agent_num]["next"]["done"].copy_(False)
                    experience_td[obs_number, agent_num]["next"]["terminated"].copy_(False)

                    experience_td[obs_number, agent_num]["hidden_states"].copy_(current_agent_data["hidden_states"])
                    experience_td[obs_number, agent_num]["next"]["hidden_states"].copy_(current_agent_data["next", "hidden_states"])
                    
                    current_agent_data["observation"] = current_obs
                    current_agent_data["hidden_states"].copy_(current_agent_data["next", "hidden_states"])

                actions = current_agent_data["action"]
                data.act.k_min_out = actions[0].item()
                data.act.k_max_out = actions[0].item() + actions[1].item()
                data.act.p_max_out = actions[2].item()

                agent_obs_num[agent_num] += 1

            full_batch_collected = all([batch_size == agent_obs_num[agent_num] for agent_num in range(num_agents)])
            if full_batch_collected:
                collection_barrier.wait()
                print("FULL BATCH COLLECTED")
                training_barrier.wait()
                agent_obs_num = {agent_num: 0 for agent_num in range(num_agents)}
                for i in range(num_agents):
                    update_model_weights_from_shared_memory(local_policies[i], local_values[i], models_td, i, device)

                agent_obs_data = TensorDict({
                    "observation": torch.zeros(num_agents, NUM_STATE_PARAMS),
                    "action": torch.zeros(num_agents, NUM_ACTIONS),
                    "next_obs": torch.zeros(num_agents, NUM_STATE_PARAMS),
                    "is_first_obs": torch.ones(num_agents, dtype=torch.bool),
                    "hidden_states": torch.zeros(num_agents, NUM_HIDDEN_CELLS)
                }, batch_size=[num_agents], device=device)
                print("STARTING DATA COLLECTION")
        print("FULL BATCH COLLECTED")
    except Exception as e:
        print(e)
    finally:
        py_interface.FreeMemory()

def create_value_module():
    value_net = nn.Sequential(
        nn.Linear(NUM_STATE_PARAMS, NUM_HIDDEN_CELLS),
        nn.Tanh(),
        nn.Linear(NUM_HIDDEN_CELLS, NUM_HIDDEN_CELLS),
        nn.Tanh(),
        nn.Linear(NUM_HIDDEN_CELLS, 1)
    )
    return ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

def create_policy_module(device):
    low = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device) #.expand(*self.batch_size, NUM_ACTIONS)
    high = torch.tensor([1000.0, 3000.0, 1.0], dtype=torch.float32, device=device)# .expand(*self.batch_size, NUM_ACTIONS)

    action_spec = Bounded(
        low=low,
        high=high,  # Currently set max values of K_min and K_max to be the total buffer size.
        shape=(NUM_ACTIONS,),
        dtype=torch.float32,
        device=device
    )

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

    return ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={  # Ensure the distributions are between the low and high
            "low": action_spec.space.low, 
            "high": action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    batch_size = FRAMES_PER_BATCH
    num_agents = 5

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    policy_module = create_policy_module(device)

    policies = [create_policy_module(device) for _ in range(num_agents)]

    value_module = create_value_module()

    value_modules = [create_value_module() for _ in range(num_agents)]

    policy_params, policy_buffers = stack_module_state(policies)
    value_params, value_buffers = stack_module_state(value_modules)

    experience_td = TensorDict({
        "observation": torch.zeros(batch_size, num_agents, NUM_STATE_PARAMS),
        "action": torch.zeros(batch_size, num_agents, NUM_ACTIONS),
        "loc": torch.zeros(batch_size, num_agents, NUM_ACTIONS),
        "scale": torch.zeros(batch_size, num_agents, NUM_ACTIONS),
        "action_log_prob": torch.zeros(batch_size, num_agents),
        "hidden_states": torch.zeros(batch_size, num_agents, NUM_HIDDEN_CELLS),
        # TODO: The below should be removed once the policy is actually ran in the data collection.
        "raw_output": torch.zeros(batch_size, num_agents, 2 * NUM_ACTIONS),

        "next": TensorDict({
            "reward": torch.zeros(batch_size, num_agents),
            "done": torch.zeros(batch_size, num_agents, dtype=torch.bool),
            "terminated": torch.zeros(batch_size, num_agents, dtype=torch.bool),
            "observation": torch.zeros(batch_size, num_agents, NUM_STATE_PARAMS),
            "hidden_states": torch.zeros(batch_size, num_agents, NUM_HIDDEN_CELLS)
        })
    }, batch_size=[batch_size, num_agents])

    models_td = TensorDict({
        "policy_params": policy_params,
        "policy_buffers": policy_buffers,
        "value_params": value_params,
        "value_buffers": value_buffers
    }, batch_size=[num_agents])

    experience_td.share_memory_()
    models_td.share_memory_()

    collection_barrier = mp.Barrier(num_agents + 1)
    training_barrier = mp.Barrier(num_agents + 1)
    save_trained_params_barrier = mp.Barrier(num_agents)


    agent_args = [(i, experience_td, models_td, collection_barrier, training_barrier) for i in range(num_agents)]

    collection_process = mp.Process(target=do_data_collection, args=(experience_td, models_td, batch_size, num_agents, collection_barrier, training_barrier))
    
    agent_processes = []
    for i in range(num_agents):
        p = mp.Process(target=do_work, args=(i, experience_td, models_td, collection_barrier, training_barrier, save_trained_params_barrier))
        agent_processes.append(p)

    try:
        collection_process.start()
        for p in agent_processes:
            p.start()
        
        collection_process.join()
        for p in agent_processes:
            p.join()
    except KeyboardInterrupt:
        collection_process.terminate()

