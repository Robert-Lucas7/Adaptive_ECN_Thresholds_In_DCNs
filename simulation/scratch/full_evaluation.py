import subprocess
import os
import time
import torch

def cleanup_ipc():
    """
    Forcefully removes shared memory segments (-M) and semaphores (-S) 
    associated with the ns3-ai interface IDs to prevent lock errors.
    We iterate through a small range in case sim_num incremented the IDs.
    """
    print("Cleaning up shared memory and semaphores...")
    for i in range(5): # Cleans up base IDs + offsets
        # Remove shared memory
        os.system(f"ipcrm -M {1234 + i} 2>/dev/null")
        os.system(f"ipcrm -M {2333 + i} 2>/dev/null")
        # Remove semaphores (this fixes the lock error)
        os.system(f"ipcrm -S {1234 + i} 2>/dev/null")
        os.system(f"ipcrm -S {2333 + i} 2>/dev/null")
        
    # Optional: ensure no zombie ns3 processes are holding locks
    os.system("pkill -9 -f run.py 2>/dev/null")
    os.system("pkill -9 -f waf 2>/dev/null")
    time.sleep(1)

hidden_layers = [1,2,3]
USE_SNN = True

def get_saved_models_suffix(hidden_layer):
    return f"{'snn' if USE_SNN else 'ann'}_{hidden_layer}_hidden_layers_{64}_hidden_cells_{'DIRECT'}_{'DIRECT'}"

def train_and_eval_configurations():
    # Make a simulation results directory (unique with the current timestamp)
    results_dir = f"./mix/{int(time.time())}"
    os.mkdir(results_dir)

    for hidden_layer in hidden_layers:
        params = {
            "--network_type": "snn",
            "--hidden_layers": hidden_layer,
        }
        # Check if the 'saved_models' directory exists for the current configuration.
        saved_models_fp = f"../saved_models_{get_saved_models_suffix(hidden_layer)}"

        # if os.path.exists(saved_models_fp):
        #     raise Exception("Saved_models directory already exists!")
        #     # saved_models_fp = f'{saved_models_fp}_new'
        #     # print(f"Directory Already Exists - creating new directory ({saved_models_fp})", flush=True)
        #     # if os.path.exists(saved_models_fp):
        #     #     raise Exception("New directory already exists! Please maintain the directories.")
        #     # else:
        #     #     os.mkdir(saved_models_fp)
        # else:
        #     os.mkdir(saved_models_fp)

        script_args = [f"{key}={value}" for key, value in params.items()]

        cleanup_ipc()

        # Train the RL agent with the network specified.
        subprocess.run(["python", "scratch/sync-collector.py", *script_args], check=True)

        cleanup_ipc()

        # Evaluate the newly trained model
        subprocess.run(["python", "scratch/sync-collector.py", "--eval", *script_args], check=True)

        # The simulation results now exist in the 'mix/' directory - read from them.
        rl_instant_qlen_fp = "./mix/instant_qlen_star_5_web_search_5_80_0.1s_dcqcn_1_.txt"
        
        os.rename(rl_instant_qlen_fp, f'{results_dir}/instant_qlen_{get_saved_models_suffix(hidden_layer)}.txt')

def generate_reward_data(hidden_layer):
    saved_models_dir = f"../saved_models_{get_saved_models_suffix(hidden_layer)}"
    
    # The batch increments must match the code in sync-collector.py
    batch_increment = 5
    end_batch = 40
    NUM_AGENTS = 5

    for batch in range(0, end_batch, batch_increment):
        args = [
            "--network_type", "snn",
            "--hidden_layers", str(hidden_layer),
            "--batch_offset", str(batch),
            "--eval",
        ]
        subprocess.run(["python", "scratch/sync-collector.py", *args])  # TODO: The evaluation must return the accumlated reward.



train_and_eval_configurations()