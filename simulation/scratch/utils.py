def get_instantaneous_data(fp):
    all_data = []
    with open(fp, 'r') as f:
        all_lines = f.readlines()
        cur_timestep_data = {}
        cur_time = None
        for i, line in enumerate(all_lines):
            elems = line.strip().split(' ')
            if len(elems) == 1:
                if cur_time is not None:
                    all_data.append({
                        cur_time: cur_timestep_data
                    })
                    cur_timestep_data = {}
                cur_time = int(elems[0])
                agent_num = 0
            else:
                bw, txRate, averageQLength, txRateECN, k_min, k_delta, p_max = [float(x) for x in elems]
                cur_timestep_data[agent_num] = {
                    "bw": bw,
                    "txRate": txRate,
                    "averageQLength": averageQLength,
                    "txRateECN": txRateECN,
                    "k_min": k_min,
                    "k_delta": k_delta,
                    "p_max": p_max,
                }
                agent_num += 1

                # Ensure the last timesteps data is added
                if i == len(all_lines) - 1:
                    all_data.append({
                        cur_time: cur_timestep_data
                    })
    # print(len(all_data))
    return all_data 

def calculate_reward(timestep_data, w, alpha):
    # print(timestep_data, flush=True)
    avg_q_len = timestep_data["averageQLength"]
    txRate = timestep_data["txRate"]

    n_power = 0
    for n in range(1, 10):
        n_power = n
        if alpha * 2**n > (avg_q_len/1000):
            break

    T = txRate / (100_000_000_000 / 8.0)
    D = 1 - n_power / 10
    # print(f'txRate: {txRate}, avg_q_len: {avg_q_len}, weighted txRate: {self.w * txRate}, weighted avg_q_len: {(1-self.w) * avg_q_len}')
    # print(f'T: {T}, D: {D}, weighted T: {self.w * T}, weighted D: {(1-self.w) * D}')
    r = w * T + (1 - w) * D
    return {
        "reward": r,
        "T": T,
        "D": D,
        "n": n_power
    }

def get_calculated_data(instant_data, port, w, alpha):
    agent_data =  [[(t, d[port]) for t, d in x.items()][0] for x in instant_data]
    times = [x[0] for x in agent_data]
    calculated_data = {
        "throughput": [x[1]["txRate"] for x in agent_data],
        "queue_length": [x[1]["averageQLength"] for x in agent_data],
        "reward": [calculate_reward(x[1], w, alpha)["reward"] for x in agent_data],
        "acc_reward": 0,
        "thresholds": {
            "k_min": [x[1]["k_min"] for x in agent_data],
            "k_max": [x[1]["k_min"] + x[1]["k_delta"] for x in agent_data]
        },
        "p_max": [x[1]["p_max"] for x in agent_data],
        "n_power": [calculate_reward(x[1], w, alpha)["n"] for x in agent_data],
        "times": times
    }
    
    calculated_data["acc_reward"] = sum(calculated_data["reward"])

    return calculated_data

def get_total_pfc_pause(fp):
    with open(fp, 'r') as f:
        total_pause_ns = 0
        active_pauses = {}

        lines = f.readlines()
        for line in lines:
            # print(line, flush=True)
            parts = line.strip().split()
            if len(parts) == 5:
                ts = int(parts[0])
                port = parts[1]
                state = parts[4]
                
                if state == '1':
                    # Record the start time for this specific port
                    active_pauses[port] = ts
                elif state == '0':
                    # Calculate duration and remove from active list
                    duration = ts - active_pauses.pop(port)
                    total_pause_ns += duration
        return total_pause_ns