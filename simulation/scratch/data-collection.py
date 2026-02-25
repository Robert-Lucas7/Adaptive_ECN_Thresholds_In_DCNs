from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import py_interface
from ctypes import Structure, c_double, c_uint64
import multiprocessing

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

def do_work(agent_num, td, collection_barrier, training_barrier):
    # An individual RL agent - used for training.
    # print(td[:, agent_num])
    num_batches_processed = 0
    while True:
        print(f'{agent_num}: Training process started...')
        collection_barrier.wait()
        print("f{agent_num} is training")
        training_barrier.wait()


def do_data_collection(td, batch_size, num_agents, collection_barrier, training_barrier):
    try:
        print("Collecting data...")
        py_interface.Init(1234, 4096)
        rl = py_interface.Ns3AIRL(2333, Env, Act)
        
        # Start simulation here - The simulation allocates the memory for the IPC.
        # rl = py_interface.Ns3AIRL(2333, Env, Act)
        agent_obs_num = {agent_num: 0 for agent_num in range(num_agents)}
        is_fork = multiprocessing.get_start_method() == "fork"
        device = (
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )
        
        while not (rl.isFinish()):
            with rl as data:
                if data is None:
                    # Simulation has finished for the specified trace file - restart it or generate new trace and run new simulation.
                    print("DATA IS NONE")
                    pass

                agent_num = data.env.agentNum
                # print(f'Agent num: {agent_num}')
                if agent_obs_num[agent_num] >= batch_size:
                    data.act.k_min_out = data.env.k_min_in
                    data.act.k_max_out = data.env.k_min_in
                    data.act.p_max_out = data.env.p_max_in
                    continue
                # print(td[:, agent_num])
                td[:, agent_num]["observation"][agent_obs_num[agent_num]] = torch.tensor([
                    data.env.BW / 1e10,              # Scale by max expected bandwidth
                    data.env.txRate / 1e10, 
                    data.env.averageQLength / 1000,  # Scale by typical max queue
                    data.env.txRateECN / 1e10,
                    data.env.k_min_in / 1000,
                    data.env.k_max_in / 1000,
                    data.env.p_max_in
                ], device=device, dtype=torch.float32)

                data.act.k_min_out = 400.0
                data.act.k_max_out = 1600.0
                data.act.p_max_out = 0.2

                agent_obs_num[agent_num] += 1
            full_batch_collected = all([batch_size == agent_obs_num[agent_num] for agent_num in range(num_agents)])
            if full_batch_collected:
                collection_barrier.wait()
                print("FULL BATCH COLLECTED")
                training_barrier.wait()
                agent_obs_num = {agent_num: 0 for agent_num in range(num_agents)}
                print("STARTING DATA COLLECTION")
        print("FULL BATCH COLLECTED")
    except Exception as e:
        print(e)
    finally:
        py_interface.FreeMemory()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    batch_size = 64
    num_agents = 5
    NUM_ACTIONS = 3
    NUM_STATE_PARAMS = 7

    td = TensorDict({
        "observation": torch.zeros(batch_size, num_agents, NUM_STATE_PARAMS),
        "action": torch.zeros(batch_size, num_agents, NUM_ACTIONS),
        "reward": torch.zeros(batch_size, num_agents),
    }, batch_size=[batch_size, num_agents])

    td.share_memory_()

    collection_barrier = mp.Barrier(num_agents + 1)
    training_barrier = mp.Barrier(num_agents + 1)

    agent_args = [(i, td, collection_barrier, training_barrier) for i in range(num_agents)]

    collection_process = mp.Process(target=do_data_collection, args=(td, batch_size, num_agents, collection_barrier, training_barrier))
    
    agent_processes = []
    for i in range(num_agents):
        p = mp.Process(target=do_work, args=(i, td, collection_barrier, training_barrier))
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

