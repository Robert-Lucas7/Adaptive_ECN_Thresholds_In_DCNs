import torch
import matplotlib.pyplot as plt

# TODO: pass device as an input to use for all tensors in this class.
class DeltaModulator():
    # Use a sliding window to calculate the spike density and then adjust the thresholds to reach this target.
    # Delta modulation encoder - decoupled from the spiking neural network to ensure a fairer comparison.
    
    def __init__(self, init_thresholds, val_names, learning_rate, window_size, spike_density, saved_models_suffix, fix_thresholds=False, NUM_PORTS=256, NUM_FEATURES=7):
        if len(init_thresholds) != len(val_names):
            raise Exception("init_thresholds and val_names must be the same length.")
        
        self.thresholds = init_thresholds

        self.lr = learning_rate
        self.win_size = window_size
        self.desired_spike_density = spike_density

        # self.spikes_buffer = torch.zeros((self.win_size, NUM_PORTS, NUM_FEATURES), device='cuda:0')
        # self.cur_vals = None
        # self.prev_vals = None
        self.count = 0 # How many times the encode method has been called - to determine whether the thresholds should be updated
        self.fix_thresholds = fix_thresholds

        self.spike_rate_hist = []
        self.saved_models_suffix = saved_models_suffix
        self.threshold_hist = []


    def update_thresholds(self, mask=None):
        if self.fix_thresholds:  # Do not change thresholds if they should be fixed.
            return

        # 1. Use absolute values: Both +1 and -1 are valid spikes in Delta Modulation.
        # If we don't use .abs(), positive and negative spikes will cancel each other out!
        abs_spikes = self.spikes_buffer.abs()

        # 2. Calculate valid elements and masked spike sum
        if mask is not None:
            # Ensure mask is broadcastable: shape [port, 1]
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            
            # Apply mask across the port dimension, then sum over win_size (dim=0) and ports (dim=1)
            # This yields a 1D tensor of shape [features]
            valid_spike_sum = (abs_spikes * mask.unsqueeze(0)).sum(dim=(0, 1))
            
            # Total valid observation slots: win_size * number of active ports
            active_ports = mask.sum()
            total_valid_elements = self.win_size * active_ports
        else:
            raise Exception("A mask must be used in update_thresholds to only count the active ports!")

        # 3. Guard against completely empty environments
        if total_valid_elements == 0:
            return

        # 4. Calculate actual density per feature (Shape: [features])
        spike_densities = valid_spike_sum / total_valid_elements

        self.spike_rate_hist.append(spike_densities)
        self.threshold_hist.append(self.thresholds.clone())

        # Print outputs matching your original code
        print(f'Current thresholds: {self.thresholds.tolist()}', flush=True)
        print(f'Spike densities: {spike_densities.tolist()}, target: {self.desired_spike_density}', flush=True)

        # 5. Safe Vectorized Update
        # Clamp densities to avoid a Division by Zero crash if a feature had exactly 0 spikes in the window
        safe_densities = spike_densities.clamp(min=1e-8)

        # Apply your exact multiplicative logic in one vectorized mathematical sweep
        self.thresholds = self.thresholds * (1.0 + self.lr * ((safe_densities / self.desired_spike_density) - 1.0))
        
        # (Optional but recommended) Prevent thresholds from collapsing to exactly 0, 
        # which would cause an infinite spike loop on the next forward pass.
        self.thresholds = self.thresholds.clamp(min=1e-6)
    
    def encode(self, cur_vals, mask, thresholds=None):
        # TODO: need to check the shape of cur_vals - whether the signal is being encoded from forward_step or forward_sequences.
        # A single observation - tensor of shape NUM_FEATURES
        if cur_vals.dim() < 4:  # forward_step
            # Lazy Initialization OR Re-initialization if NS-3 changes active ports
            if getattr(self, 'prev_vals', None) is None or self.prev_vals.shape != cur_vals.shape:
                self.prev_vals = cur_vals.clone()
                
                # Dynamically construct the ring buffer shape: [win_size, *cur_vals_shape]
                # If cur_vals is [5, 7], buffer_shape becomes (2000, 5, 7)
                buffer_shape = (self.win_size,) + cur_vals.shape
                
                # Create the buffer exactly matching the environment's current dimensions
                self.spikes_buffer = torch.zeros(buffer_shape, dtype=torch.float32, device=cur_vals.device)
                self.count = 0
                
                return torch.zeros_like(cur_vals)
            
            # self.cur_vals has been populated already, so a difference can be calculated.
            self.cur_vals = cur_vals

            deltas = self.cur_vals - self.prev_vals
            
            self.prev_vals = self.cur_vals.clone()

            pos_spikes = (deltas > self.thresholds).float()
            neg_spikes = (deltas < -self.thresholds).float() * -1.0
    
            current_spikes = pos_spikes + neg_spikes
            current_spikes = current_spikes * mask
            
            buffer_idx = self.count % self.win_size
            self.spikes_buffer[buffer_idx] = current_spikes

            if (self.count + 1) % self.win_size == 0:
                self.update_thresholds(mask)

            self.count += 1

            return current_spikes
        else:
            if thresholds is None:
                raise Exception("Thresholds must be passed into 'encode' method for forward_sequences so the signal reconstruction is the same as in the rollout.")
            # Encode a series of sequences with delta modulation
            # spikes = torch.zeros(cur_vals.size(), device='cuda:0')  # Same shape - just a spiking representation
            # for batch in range(cur_vals.size(0)):
            #     prev_vals = cur_vals[batch, 0, :]
            #     for t in range(1, cur_vals.size(1)):
            #         diff = cur_vals[batch, t, :] - prev_vals
            #         for i, delta in enumerate(diff):
            #             if delta > self.thresholds[i]:
            #                 spikes[batch, t, i] = 1
            #             else:
            #                 spikes[batch, t, i] = 0
            #         prev_vals = cur_vals[batch, t, :]
            # return spikes
            # Compute differences between consecutive timesteps across all batches/features simultaneously
            diff = cur_vals[:, 1:, ...] - cur_vals[:, :-1, ...]  # Shape: [batch, time-1, features]

            sliced_thresholds = thresholds[:, 1:, ...]

            # 3. Calculate BOTH positive and negative spikes against the sliced thresholds
            pos_spikes = (diff > sliced_thresholds).float()
            neg_spikes = (diff < -sliced_thresholds).float() * -1.0
            
            # 4. Create the empty 16-step spike tensor
            spikes = torch.zeros_like(cur_vals)
            
            # 5. Insert the 15 steps of calculated spikes into slots 1 through 15
            spikes[:, 1:, ...] = pos_spikes + neg_spikes
            spikes = spikes * mask
            
            assert not torch.isnan(spikes).any(), "NaN in spike input"
            assert not torch.isinf(spikes).any(), "Inf in spike input"

            return spikes
    
if __name__ == "__main__":
    # ================== Some testing ... ===================
    dm = DeltaModulator([4], ["averageQLen"], 0.1, 10, 0.5)
    test_data = [1,2,3,8,3,4,5,3,2,10,7,6,5,4,6,10,15,13,12,11,10]

    for val in test_data:
        print(dm.encode({"averageQLen": val}))
        dm.update_thresholds()
        print(dm.thresholds)