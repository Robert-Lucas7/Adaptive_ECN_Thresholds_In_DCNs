import sys
sys.dont_write_bytecode = True  # Prevent __pycache__ directory being created.

import subprocess
import os
import time
import torch
from utils import get_instantaneous_data, get_calculated_data
import matplotlib.pyplot as plt
import json
import multiprocessing


def cleanup_ipc(sim_num):
    """
    Forcefully removes shared memory segments (-M) and semaphores (-S) 
    associated with the ns3-ai interface IDs to prevent lock errors.
    We iterate through a small range in case sim_num incremented the IDs.
    """
    print("Cleaning up shared memory and semaphores...")
 
    os.system(f"ipcrm -M {2333 + sim_num} 2>/dev/null")
    # Remove semaphores (this fixes the lock error)
    os.system(f"ipcrm -S {2333 + sim_num} 2>/dev/null")
    
    time.sleep(1)

# hidden_layers = [2]
betas = [0.8]
firing_rates = [0.1] # , 0.2, 0.4, 0.6, 0.8] # [0.1, 0.9, 0.4, 0.2, 0.6, 0.8] # [0.2, 0.4, 0.6, 0.8]
input_encodings = ["direct", "delta"] #["direct", "delta", "rate"]  # ["delta", "direct", "rate"]
cont_or_discs = ["DISCRETE"] #, "CONTINUOUS"]
mse_scaling = [30] # [1, 5, 10, 15, 20, 25]

USE_SNN = True


EVAL_TOPO = "FAT"  # STAR or FAT

if EVAL_TOPO == "STAR":
    EVAL_TRACE = "web_search_60_50_0.1s"
elif EVAL_TOPO == "FAT":
    EVAL_TRACE = "web_search_fat_30_0.01s"
elif EVAL_TOPO == "TESTING":
    EVAL_TRACE = "web_search_5_80_0.1s"
else:
    raise Exception("EVAL_TOPO must be either 'star' or 'fat'.")

MOMENTUM = 0.01

RL_INSTANT_QLEN_FP = "./mix/instant_qlen_{EVAL_TOPO}_{EVAL_TRACE}_{encoding}_{firing_rate}dcqcn_1_.txt"
RL_FCT_FP = "./mix/fct_{EVAL_TOPO}_{EVAL_TRACE}_1_{encoding}_{firing_rate}_dcqcn.txt"
RL_PFC_FP = "./mix/pfc_{EVAL_TOPO}_{EVAL_TRACE}__{encoding}_{firing_rate}_dcqcn.txt"

def get_saved_models_suffix(topo, hidden_layer, beta, firing_rate, cont_or_disc, enc_in, scaling=1):
    s = f"{topo}_{'snn' if USE_SNN else 'ann'}_{cont_or_disc}_{hidden_layer}_hidden_layers_{64}_hidden_cells_{enc_in.upper()}_{'DIRECT'}_BETA_{beta}_FIRING_RATE_{firing_rate}"
    if enc_in == "direct":
        s += f"_MSE_SCALING_{scaling}"
    # elif enc_in == "delta":
    #     s += f"_telem_momentum_{MOMENTUM}"
    return s

def eval_config(params, saved_models_suffix, results_dir, file):
    
    print(f"Evaluation with params; {saved_models_suffix}", flush=True)
    script_args = [f"--{key}={value}" for key, value in params.items()]

    # Evaluate the newly trained model
    subprocess.run(["python", "scratch/sync-collector.py", "--eval", *script_args], check=True, stdout=file, stderr=file)

def train_and_eval_configurations(params, results_dir, do_training=True, do_eval=False):
    script_args = [f"--{key}={value}" for key, value in params.items()]

    saved_models_suffix = get_saved_models_suffix(
        EVAL_TOPO,
        hidden_layer=params["hidden_layers"],
        beta=params["beta"],
        firing_rate=params["firing_rate"],
        cont_or_disc=params["cont_or_disc"],
        enc_in=params["enc_in"],
        scaling=params["mse_loss_scale"]
    )

    sim_num = params['sim_num']

    print(f"Starting simulation with sim_num = {sim_num}", flush=True)

    if do_training:
        # Train the RL agent with the network specified.
        cleanup_ipc(sim_num)

        with open(f"../training_logs/{saved_models_suffix}.log", "w") as f:
            subprocess.run(["python", "scratch/sync-collector.py", *script_args], check=True, stdout=f, stderr=f)

    if do_eval:
        cleanup_ipc(sim_num)

        with open(f'../eval_logs/{saved_models_suffix}.log', 'w') as f:
            eval_config(params, saved_models_suffix, results_dir, f)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    results_dir = f"./mix/{int(time.time())}"
    os.mkdir(results_dir)

    sim_num = 0
    sim_num_offset = 450  # If other trainings are still running, use sim_num_offset so there are no conflicts.
    processes = []
    for i, firing_rate in enumerate(firing_rates):
        
        for enc_in in input_encodings:
            scalings = mse_scaling if enc_in == "direct" else [1]
            for scaling in scalings:
                for beta in betas:
                    num_layers = 2 if enc_in == "direct" else 1

                    params = {
                        "network_type": "snn",
                        "hidden_layers": num_layers,
                        "enc_in": enc_in,
                        "enc_out": "direct",
                        "beta": beta,
                        "firing_rate": firing_rate,
                        "cont_or_disc": "DISCRETE",
                        "mse_loss_scale": scaling,
                        "sim_num": sim_num + sim_num_offset,
                        "topo": EVAL_TOPO,
                        "batch_offset": 70
                    }

                    p = multiprocessing.Process(target=train_and_eval_configurations, args=(params, results_dir, False, True, ))
                    p.start()
                    processes.append(p)

                    sim_num += 1

    for p in processes:
        p.join()
                        
