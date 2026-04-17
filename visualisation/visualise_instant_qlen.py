import subprocess
import matplotlib.pyplot as plt
# import pandas as pd
import json
from collections import Counter
import numpy as np

axes_font_size = 24
legend_font_size = 18

most_recent_rl = "../../simulation/mix/instant_qlen_star_5_{trace}_dcqcn_1_.txt"
most_recent_static = "../../simulation/mix/instant_qlen_star_5_{trace}_dcqcn_0_.txt"

default_web_search_0_1s = "../simulation/mix/instant_qlen_default_web_search_0.1s.txt"
ga_web_search_0_1s = "../simulation/mix/instant_qlen_ga_web_search_0.1s.txt"

data_info = [
{
    "label": "RL - ANN - 2 hidden layer",
    "fp": most_recent_rl.format(trace="web_search_5_80_0.1s")
},
{
    "label": "Static - default",
    "fp": most_recent_static.format(trace="web_search_5_80_0.1s")
}

# {
#     "label": "Aggressive ECN parameters",
#     "fp": "../../simulation/mix/instant_qlen_star_5_agressive_static.txt"
# },
# {
#     "label": "Relaxed ECN parameters",
#     "fp": "../../simulation/mix/instant_qlen_star_5_relaxed_static.txt"
# },

# {
#     "label": "RL",
#     "fp": "../../simulation/mix/instant_qlen_star_5_{trace}_dcqcn_0_GA.txt".format(trace="web_search_5_80_0.1s")
# }
]

# for h in [1,2,3]:
#     data_info.append({
#         "label": f"RL - SNN - {h} hidden layers",
#         "fp": f"../../simulation/mix/varying_hidden_layers_40_batches/instant_qlen_snn_{h}_hidden_layers_64_hidden_cells_DIRECT_DIRECT.txt"
#     })

REWARD_W = 0.1
TO_PLOT = [] # , "Sending Rate"]
N_HOSTS = 5
PORT = 1


# TODO: move/modify script to import this function from utils so there is only one instance of it.

def get_instantaneous_data(fp):
    """
    Parse the data from an "instant_qlen" file generated from the NS3 simulation into
    a suitable python object with the format:
    [
        time: {
            agent1: {
                port1: {
                    "bw": bw,
                    "txRate": txRate,
                    "averageQLength": averageQLength,
                    etc
                },
                portN: {...}
            },
            agentN: {...}
        },
        timeN: {...}
    ]
    """
    all_data = []
    with open(fp, 'r') as f:
        all_lines = f.readlines()
        cur_timestep_data = {}
        cur_time = None
        port_num = 0
        for i, line in enumerate(all_lines):
            if line[:4].upper() == "TIME":
                if cur_time is not None:
                    all_data.append({
                        cur_time: cur_timestep_data
                    })
                    cur_timestep_data = {}
                cur_time = int(line.split()[1].strip())
            elif line[:5].upper() == "AGENT":
                agent_num = int(line.split()[1].strip())
                cur_timestep_data[agent_num] = {}
                port_num = 0
            else:
                elems = line.strip().split(' ')
                bw, txRate, averageQLength, txRateECN, k_min, k_delta, p_max = [float(x) for x in elems]
                cur_timestep_data[agent_num][port_num] = {
                    "bw": bw,
                    "txRate": txRate,
                    "averageQLength": averageQLength,
                    "txRateECN": txRateECN,
                    "k_min": k_min,
                    "k_delta": k_delta,
                    "p_max": p_max,
                }
                port_num += 1
            # Ensure the last timesteps data is added
            if i == len(all_lines) - 1:
                all_data.append({
                    cur_time: cur_timestep_data
                })
    print(len(all_data))
    return all_data #  [:500] #[:20000]

AGENT = 0
PORT = 0

data = [
        {
            "label": info["label"],
            "data": get_instantaneous_data(info["fp"])[1:]
        }
        for info in data_info
    ]
# print(json.dumps(data, indent=4))

# print(list(data[0].values())[0][AGENT])
# Comparing rewards for the static and RL implementations.
# N.B This function should be changed to reflect the one in RLEnv._calculate_reward()
def calculate_reward(timestep_data, w, alpha):
    avg_q_len = timestep_data["averageQLength"]
    txRate = timestep_data["txRate"]

    n_power = 0
    for n in range(1, 10):
        n_power = n
        if alpha * 2**n > (avg_q_len/1000):
            break

    T = txRate / (100_000_000_000 / 8.0)  # Link utilisation.
    # D = 1 - n_power / 10
    # print(f'txRate: {txRate}, avg_q_len: {avg_q_len}, weighted txRate: {self.w * txRate}, weighted avg_q_len: {(1-self.w) * avg_q_len}')
    # print(f'T: {T}, D: {D}, weighted T: {self.w * T}, weighted D: {(1-self.w) * D}')
    # r = w * T + (1 - w) * D

    T = txRate / 100_000_000_000
    L = 1 / max(0.000001, avg_q_len)
    # print(f'txRate: {txRate}, avg_q_len: {avg_q_len}, weighted txRate: {self.w * txRate}, weighted avg_q_len: {(1-self.w) * avg_q_len}')
    # print(f'T: {T}, D: {D}, weighted T: {self.w * T}, weighted D: {(1-self.w) * D}')
    # r = self.w * T + (1-self.w) * D

    # r = -avg_q_len
    w = 0.5
    r = w * T + (1 - w) * L
    

    return {
        "reward": r,
        "T": T,
        # "D": D,
        # "n": n_power
    }

def make_plots(w, alpha):
    colours = ["orange", "blue", "green"]
    for end_time in [2100000000, 2010000000]:
        end_idx = len(data[0]['data']) - 1
        # print(len(data[0]['data']))
        for idx, entry in enumerate(data[0]['data']):
            # print(json.dumps(entry, indent=4))
            time = list(entry.keys())[0]
            # print(time)
            if time > end_time:
                end_idx = idx
                break
        # print(f'End time: {time}')
        # print(f'End index: {end_idx}')

        x_axis_label = "Simulation Time (seconds)"
        plot_details = [
            {
                "name": "throughput",
                "title": "Throughput for aggressive vs relaxed ECN parameters",
                "y_axis_label": "Throughput (Bytes)"
            },
            {
                "name": "queue_length",
                "title": "Queue Length for aggressive vs relaxed ECN parameters",
                "y_axis_label": "Queue Length (Bytes)"
            }, {
                "name": "reward",
                "title": "Reward for static vs adaptive ECN thresholds",
                "y_axis_label": "Reward"
            }, {
                "name": "acc_reward",
                "title": "Accumulated reward",
                "y_axis_label": "Reward"
            }, {
                "name": "thresholds",
                "title": "ECN thresholds for static vs adaptive thresholds",
                "y_axis_label": "Threshold Value (Bytes)"
            }, {
                "name": "p_max",
                "title": "P_max for static vs adaptive thresholds",
                "y_axis_label": "P_max"
            }
            # , {
            #     "name": "n_power",
            #     "title": "n_power",
            #     "y_axis_label": "n"
            # }
        ]
        plt.rcParams.update({'font.size': axes_font_size})
        for detail in plot_details:
            fig, ax = plt.subplots(figsize=(10,8))
            for i, item in enumerate(data):
                # agent_data =  [list(x.values())[0][AGENT] for x in item["data"]]
                # print(json.dumps(item["data"], indent=4))
                agent_data =  [[(t, d[AGENT][PORT]) for t, d in x.items()][0] for x in item["data"][:end_idx]]
                times = [x[0] for x in agent_data]

                calculated_data = {
                    "throughput": [x[1]["txRate"] for x in agent_data],
                    "queue_length": [x[1]["averageQLength"] for x in agent_data],
                    "reward": [calculate_reward(x[1], w, alpha)["reward"] for x in agent_data],
                    "acc_reward": [],
                    "thresholds": {
                        "k_min": [x[1]["k_min"] for x in agent_data],
                        "k_max": [x[1]["k_min"] + x[1]["k_delta"] for x in agent_data]
                    },
                    "p_max": [x[1]["p_max"] for x in agent_data]
                    # "n_power": [calculate_reward(x[1], w, alpha)["n"] for x in agent_data]
                }

                reward_steps = 100
                for t_r in range(0, len(calculated_data["reward"]), reward_steps):
                    calculated_data["acc_reward"].append(sum(calculated_data["reward"][t_r:t_r+reward_steps]))



                # Plot the results
                

                if detail['name'] == "thresholds":
                    # print(f'{item["label"]}: {calculated_data[detail["name"]]["k_min"]}')
                    # print(calculated_data[detail['name']]['k_max'])
                    ax.plot(times, calculated_data[detail['name']]['k_min'], label=f'{item["label"]} - k_min', color=colours[i])
                    ax.plot(times, calculated_data[detail['name']]['k_max'], label=f'{item["label"]} - k_max', color=colours[i])
                elif detail['name'] == "acc_reward":
                    new_times = [x for x in range(0, len(calculated_data["reward"]), reward_steps)]
                    print(new_times)
                    ax.plot(new_times, calculated_data[detail['name']], label=item['label'])
                else:
                    ax.plot(times, calculated_data[detail['name']], label=item['label'])
                

            ax.set_ylabel(detail["y_axis_label"])
            ax.set_xlabel("Simulation Time (seconds)")
            plt.rcParams.update({'font.size': legend_font_size})
            ax.legend(frameon=False)
            plt.rcParams.update({'font.size': axes_font_size})
            # ax.set_title(detail["title"]) # ECN thresholds with w=0.1 for reward=w * T + (1-w) * D(L)")#Queue Length with static ECN vs adaptive with ANN") # Reward with w=0.1 and r=w * T + (1-w) * D(L)")
            fig.tight_layout()
            fig.savefig(f'./{end_time}_{detail["name"]}_w_{w}_alpha_{alpha}_.png')
            plt.close(fig)

def plot_D_component(w, alpha, data_item=None):
    if data_item:
        agent_data =  [[(t, d[AGENT]) for t, d in x.items()][0] for x in data_item["data"]]
        queue_lengths = [x[1]["averageQLength"] for x in agent_data]
    else:
        queue_lengths = [x for x in range(0, 14_000_000, 1_000)]

    # powers = []
    # D = []
    # for avg_q_len in queue_lengths:
    #     n_power = 0
    #     for n in range(0, 10):
    #         n_power = n
    #         if alpha * 2**n > (avg_q_len/1000):
    #             break
    #     powers.append(n_power)
    #     D.append(1 - n_power / 10)
    # # print(item['label'], Counter(powers))
    # print(Counter(D))
    D = []
    for avg_q_len in queue_lengths:
        D.append(max(0.0, 1 - avg_q_len / 250_000.0))
    plt.plot(D) #, label=item['label'])
    plt.ylabel("D(L)")
    plt.xlabel("Queue Length (KB)")
    plt.title("D(L) vs Queue Length")
    
    if data_item:
        file_suffix = f"_{data_item['label']}"
    else:
        file_suffix = ""

    # plt.savefig(f'D_reward_component_alpha_{alpha}{file_suffix}.png')
    plt.savefig('Linear_Reward.png')
    plt.cla()
import os

def fmt_size(b):
    if b >= 1e6:
        return f"{b/1e6:.1f}MB"
    elif b >= 1e3:
        return f"{b/1e3:.1f}KB"
    return f"{int(b)}B"

colours = ["orange", "blue", "green"]
def plot_fcts(fcts):
    plt.rcParams.update({'font.size': axes_font_size})
    fct_files = [x[0] for x in fcts]
    identifiers = [x[1] for x in fcts]
    plt.cla()
    PYTHON2_PATH = "/home/links/rl624/.conda/envs/hpcc_env/bin/python"
    fig, ax = plt.subplots(figsize=(10,8))

    # ax.set_title("FCT slowdown for Agressive vs Relaxed Static ECN parameters")

    colour_idx = 0
    for idx, file in enumerate(fct_files):
        env = os.environ.copy()
        # TODO: Make this cleaner and add logging, so it will be easier to debug.
        env["PATH"] = f"{os.path.dirname(PYTHON2_PATH)}:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

        for key in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "PYTHON"]:
            env.pop(key, None)

        with open(f'../../analysis/{file}_results.txt', 'w') as f:
            print(file)
            subprocess.run(["python", "../../analysis/fct_analysis.py", "-p", f"../../simulation/mix/{file}", "-b", "25"], 
                            stdout=f, 
                            env=env)
        with open(f"../../analysis/{file}_results.txt", 'r') as f:
            lines = f.readlines()

            print(lines)
            arr = np.zeros((len(lines), 5), dtype=float)
            for i, line in enumerate(lines):
                arr[i] = np.array([float(x.strip()) for x in line.split()])
            
            steps = arr[:, 0]
            sizes = arr[:, 1]
            mid_fct = arr[:, 2]
            p95_fct = arr[:, 3]
            p99_fct = arr[:, 4]

            ax.plot(steps, mid_fct, label=f"{identifiers[idx]} - mid fct", color=colours[colour_idx])
            line = ax.plot(steps, p95_fct, label=f"{identifiers[idx]} - 95-pct fct", linestyle='--', color=colours[colour_idx])[0]
            line.set_dashes([10, 5])
            line = ax.plot(steps, p99_fct, label=f"{identifiers[idx]} - 99-pct fct", color=colours[colour_idx])[0]
            line.set_dashes([2, 2])


            ax.set_xlabel("Flow size")
            ax.set_ylabel("FCT slowdown")
            plt.rcParams.update({'font.size': legend_font_size})
            ax.legend(frameon=False)
            plt.rcParams.update({'font.size': axes_font_size})
            ax.set_xticks(steps,[fmt_size(size) for size in sizes], rotation=90)
            
            # ax.set_xticklabels([fmt_size(size) for size in sizes])

            colour_idx += 1
        fig.tight_layout()
        fig.savefig("fct_plot.png")
        plt.close()


w = 0.5
alpha=40

fcts = [
    # ("fct_star_5_agressive_static", "Aggressive"), # Static ECN,
    # ("fct_star_5_relaxed_static", "Relaxed")
    ("fct_star_5_web_search_5_80_0.1s_0", "Static"),
    ("fct_star_5_web_search_5_80_0.1s_1", "ANN")
]


plot_fcts(fcts)

# plot_D_component(w, alpha, data[0])
make_plots(w, alpha)





# all_data = [{
#     "label": info.get("label"),
#     "data": get_instantaneous_data(info.get("fp"))} for info in data_info]

# # Create figure and first axis
# fig, ax1 = plt.subplots()
# ax1.set_xlabel('Sim Time after start of flows (Nanoseconds)')
# if "Queue Length" in TO_PLOT:
    
#     ax1.set_ylabel('Queue Length (bits)')
#     for data in all_data:
#         times = [x["time"] for x in data["data"]]
#         qlens = [x["qlen"][PORT] for x in data["data"]]
#         ax1.plot(times, qlens, label=data["label"])
        
#     ax1.tick_params(axis='y')

# if "Rewards" in TO_PLOT:
#     ax1.set_ylabel('Reward')
#     for data in all_data:
#         times = [x["time"] for x in data["data"]]
#         rewards = [x["throughput"][PORT] for x in data["data"]]
#         ax1.plot(times, rewards, label=data["label"])
#     ax1.tick_params(axis='y')


# if "Sending Rate" in TO_PLOT:
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Sending rate')
#     for data in all_data:
#         times = [x["time"] for x in data["data"]]
#         send_rates = [x["send_rate"][PORT] for x in data["data"]]
#         ax2.plot(times, send_rates, label=data["label"])
#     ax2.tick_params(axis='y')

# if "Throughput" in TO_PLOT:
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Throughput (bits/ns)')
#     for data in all_data:
#         times = [x["time"] for x in data["data"]]
#         throughput = [x["throughput"][PORT] for x in data["data"]]
#         ax2.plot(times, throughput, label=data["label"])
#     ax2.tick_params(axis='y')


# # Adjust layout and show plot
# fig.tight_layout()
# # plt.show()


# # plt.plot([x[0] for x in instant_data], [x[1] for x in instant_data])
# # plt.plot([x[0] for x in instant_data], [x[2] for x in instant_data])
# plt.legend()
# plt.title("Queue Length")
# plt.savefig('instantaneous_qlen.png')