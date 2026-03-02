import matplotlib.pyplot as plt
# import pandas as pd
import json

hpcc_fp = "../../FinalYearProject/High-Precision-Congestion-Control/simulation/mix/instant_qlen_star_5_star_5_single_burst_trace_hp95.txt"
dcqcn_fp = "../simulation/mix/instant_qlen_star_5_web_search_5_80_2s_dcqcn_0.txt"
snn_fp = "../simulation/mix/instant_qlen_star_5_web_search_5_80_2s_dcqcn_1.txt"


data_info = [
#     {
#     "label": "hpcc",
#     "fp": hpcc_fp
# }, 
# {
#     "label": "dcqcn",
#     "fp": dcqcn_fp
# }
# , 
{
    "label": "SNN",
    "fp": snn_fp
}
]
TO_PLOT = ["Queue Length" ]#"Sending Rate"]


def get_instantaneous_data(fp):
    all_data = []
    with open(fp, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            t, qlen, send_rate = line.strip().split(' ')
            all_data.append((int(t), int(qlen), int(send_rate)))
    print(len(all_data))
    return all_data[:1000]

# print(get_instantaneous_data(INSTANT_QLEN_FP))

all_data = [{
    "label": info.get("label"),
    "data": get_instantaneous_data(info.get("fp"))} for info in data_info]

# Create figure and first axis
fig, ax1 = plt.subplots()
ax1.set_xlabel('Sim Time after start of flows (Nanoseconds)')
if "Queue Length" in TO_PLOT:
    
    ax1.set_ylabel('Queue Length (bits)')
    for data in all_data:
        times = [x[0] for x in data["data"]]
        qlens = [x[1] for x in data["data"]]
        ax1.plot(times, qlens, label=data["label"])
        
    ax1.tick_params(axis='y')

if "Sending Rate" in TO_PLOT:
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sending rate')
    for data in all_data:
        times = [x[0] for x in data["data"]]
        send_rates = [x[2] for x in data["data"]]
        ax2.plot(times, send_rates, label=data["label"])
    ax2.tick_params(axis='y')

# Adjust layout and show plot
fig.tight_layout()
# plt.show()


# plt.plot([x[0] for x in instant_data], [x[1] for x in instant_data])
# plt.plot([x[0] for x in instant_data], [x[2] for x in instant_data])
plt.legend()
plt.title("Queue Length")
plt.savefig('instantaneous_qlen.png')