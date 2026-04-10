import sys
sys.dont_write_bytecode = True  # Prevent __pycache__ directory being created.

import subprocess
import os
import time
import torch
from utils import get_instantaneous_data, get_calculated_data
import matplotlib.pyplot as plt
import json

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

hidden_layers = [2]
USE_SNN = True

RL_INSTANT_QLEN_FP = "./mix/instant_qlen_star_5_web_search_5_80_0.1s_dcqcn_1_.txt"


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
            "--enc_in": "delta",
            "--enc_out": "direct",
        }
        # Check if the 'saved_models' directory exists for the current configuration.
        saved_models_fp = f"../saved_models_{get_saved_models_suffix(hidden_layer)}"

        script_args = [f"{key}={value}" for key, value in params.items()]

        cleanup_ipc()

        # Train the RL agent with the network specified.
        subprocess.run(["python", "scratch/sync-collector.py", *script_args], check=True)

        cleanup_ipc()

        # Evaluate the newly trained model
        subprocess.run(["python", "scratch/sync-collector.py", "--eval", *script_args], check=True)

        # The simulation results now exist in the 'mix/' directory - read from them.
        os.rename(RL_INSTANT_QLEN_FP, f'{results_dir}/instant_qlen_{get_saved_models_suffix(hidden_layer)}.txt')


def generate_reward_data(hidden_layer, enc_in, enc_out):
    data_over_batches_dir = f'./mix/over_batches_{get_saved_models_suffix(hidden_layer)}'
    if os.path.exists(data_over_batches_dir):
        print("over batches directory already exists! Returning Existing data", flush=True)
        with open(f'{data_over_batches_dir}/{get_saved_models_suffix(hidden_layer)}', 'r') as f:
            return json.load(f)
    else:
        os.mkdir(data_over_batches_dir)

    saved_models_dir = f"../saved_models_{get_saved_models_suffix(hidden_layer)}"
    
    # The batch increments must match the code in sync-collector.py
    batch_increment = 10
    end_batch = 40
    NUM_AGENTS = 5

    model_eval_metrics = {}  # Key corresponds to the batch and the key is a dict of metrics.

    for batch in range(0, end_batch+1, batch_increment):
        args = [
            "--network_type", "snn",
            "--hidden_layers", str(hidden_layer),
            "--batch_offset", str(batch),
            "--enc_in", enc_in,
            "--enc_out", enc_out,
            "--eval",
        ]

        cleanup_ipc()

        # Running the script will generate a results file in 'RL_INSTANT_QLEN_FP' - which the accumulated rewards can be calculated from.
        subprocess.run(["python", "scratch/sync-collector.py", *args])  # TODO: The evaluation must return the accumlated reward.

        AGENT = 1  # Evaluating the agent deployed on port 1.

        instant_data = get_instantaneous_data(RL_INSTANT_QLEN_FP)

        # TODO: Move these into the arguments to 'sync-collector.py' script.
        w = 0.5
        alpha=40

        calculated_data = get_calculated_data(instant_data, AGENT, w, alpha)
        
        model_eval_metrics[batch] = calculated_data

    fig, ax = plt.subplots()
    ax.plot(list(model_eval_metrics.keys()), [x["acc_reward"] for x in model_eval_metrics.values()], label="Accumulated Reward")

    fig.savefig('accumulated_reward_over_training.png')
    with open(f'{data_over_batches_dir}/{get_saved_models_suffix(hidden_layer)}', 'w') as f:
        json.dump(model_eval_metrics, f, indent=4)

    return model_eval_metrics
    
train_and_eval_configurations()

# saved_models_snn_1_hidden_layers_64_hidden_cells_DELTA_DIRECT
# single_hidden_layer_data = generate_reward_data(hidden_layer=1, enc_in="delta", enc_out="direct")

# static_data = get_instantaneous_data("./mix/instant_qlen_static_params.txt")
# static_metrics = get_calculated_data(static_data, 1, 0.5, 40)

# fig, ax = plt.subplots()
# x = list(single_hidden_layer_data.keys())
# ax.plot(x, [x["acc_reward"] for x in single_hidden_layer_data.values()], label="SNN - 1 hidden layer")
# ax.plot(x, [static_metrics["acc_reward"]] * len(x), label="Static Params")

# ax.set_xlabel("Batch")
# ax.set_ylabel("Accumulated Reward")
# ax.legend()
# ax.set_title("Accumulated Reward - Static vs adaptive")

# fig.savefig('static_vs_adaptive_rewards.png')








# train_and_eval_configurations()