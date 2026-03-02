import matplotlib.pyplot as plt
# import pandas as pd

DCQCN_FP = "./High-Precision-Congestion-Control/simulation/mix/qlen_fat_flow_dcqcn.txt"
HPCC_FP = "./High-Precision-Congestion-Control/simulation/mix/qlen_fat_flow_hp95.txt"

"""
all_qlen_data has the format:
{
    cur_time: {
        (switch_num, port_num): [The counts where the queue was n-1 to n KB in size]
    }
}
"""
def get_qlen_data(fp):
    cur_time = None
    all_qlen_data = {}
    cur_qlen_data = {}
    with open(fp, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            if line[:4] == "time":
                if cur_time:
                    all_qlen_data[cur_time] = cur_qlen_data
                    cur_qlen_data = {}
                cur_time = int(line[6:])
            else:
                entries = line.split(' ')
                switch_and_port = [-1, -1]
                samples = []
                for idx, entry in enumerate(entries):
                    if idx == 0:
                        # Switch number
                        switch_and_port[0] = int(entry)
                    elif idx == 1:
                        # Port number
                        switch_and_port[1] = int(entry)
                    else:
                        # The number of samples with the queue length
                        samples.append(int(entry))
                cur_qlen_data[tuple(switch_and_port)] = samples
        all_qlen_data[cur_time] = cur_qlen_data
    return all_qlen_data

'''
Find the switches and ports which have had a queue length of larger than 1KB.
'''
def find_congested_ports(qlen_data):
    potential_congestion = {}
    for time_interval, interval_data in qlen_data.items():
        cur_potential_congestion = {}
        for port_and_switch, cum_samples in interval_data.items():
            if len(cum_samples) > 1:
                cur_potential_congestion[port_and_switch] = cum_samples
        if cur_potential_congestion:
            potential_congestion[time_interval] = cur_potential_congestion
    return potential_congestion


def get_switch_and_port_from_congestion(congested_ports):
    """
    Extract the switch and ports that are congested from the data structure returned from 'find_congested_ports'
    """
    switch_and_ports = set()
    for _, v in congested_ports.items():
        for s_and_p in v.keys():
            switch_and_ports.add(s_and_p)
    return switch_and_ports


def compare_two_congestion_methods(fp1, fp2, analyse=True):
    """
    Compare the qlen data between two cc methods/files produced from the simulation. 
    N.B. fp1 should be the better performing method as a difference is taken.
    analyse: bool - determines whether to analyse the two files or just output the congested switch and port numbers.
    """
    qlen_data1 = get_qlen_data(fp1)
    qlen_data2 = get_qlen_data(fp2)

    congested_port1 = find_congested_ports(qlen_data1)
    congested_port2 = find_congested_ports(qlen_data2)

    congested_switch_and_port1 = get_switch_and_port_from_congestion(congested_port1)
    congested_switch_and_port2 = get_switch_and_port_from_congestion(congested_port2)

    if (congested_switch_and_port1 != congested_switch_and_port2):
        print("=======================================================================")
        print("WARNING: The congested switch and ports differ between the two methods!")
        print("=======================================================================")
    
    if analyse:
        pass
    
    return congested_switch_and_port1, congested_switch_and_port2
    

print(compare_two_congestion_methods(HPCC_FP, DCQCN_FP))

# plt.plot([i for i in range(10)])

# plt.savefig('dummy_plot.png')