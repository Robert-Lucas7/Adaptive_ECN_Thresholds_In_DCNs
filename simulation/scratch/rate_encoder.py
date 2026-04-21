import torch
from snntorch import spikegen

class RateEncoder():
    def __init__(self, window_size, virtual_timesteps, learning_rate, spike_rate):
        self.window_size = window_size
        self.count = 0
        self.desired_spike_rate = spike_rate
        self.virtual_timesteps = virtual_timesteps
        self.scaling_factor = 1
        self.lr = learning_rate
        self.spike_rate_hist = []
        self.scaling_factor_hist = []
    def update_scaling_factor(self, mask):
        # Calculate the 

        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        
        # Apply mask across the port dimension, then sum over win_size (dim=0) and ports (dim=1)
        # This yields a 1D tensor of shape [features]
        valid_spike_sum = (self.spikes_buffer * mask.unsqueeze(0)).sum(dim=(0, 1, 2))
        
        # Total valid observation slots: win_size * number of active ports
        active_ports = mask.sum()
        total_valid_elements = self.window_size * active_ports * self.virtual_timesteps

        if total_valid_elements == 0:
            return

        # 4. Calculate actual density per feature (Shape: [features])
        spike_densities = valid_spike_sum / total_valid_elements
        print(f'Spike densities shape: {spike_densities.shape}', flush=True)
        self.spike_rate_hist.append(spike_densities)

        collective_spike_rate = spike_densities.mean().clamp(min=1e-8)
        
        # Update the global scaling factor - features lose their absolute magnitudes but maintain the relative magnitudes.
        self.scaling_factor = self.scaling_factor * (1.0 + self.lr * ((collective_spike_rate / self.desired_spike_rate) - 1.0))
        self.scaling_factor_hist.append(self.scaling_factor)


    def encode(self, cur_vals, mask, scaling_factor=None):
        if cur_vals.dim() < 4:  # forward_step
            if getattr(self, 'spikes_buffer', None) is None:
                buffer_shape = (self.window_size, self.virtual_timesteps,) + cur_vals.shape
                self.spikes_buffer = torch.zeros(buffer_shape, dtype=torch.float32, device=cur_vals.device)
                

            probs = cur_vals * self.scaling_factor * 1/self.desired_spike_rate
            spikes = spikegen.rate(probs, num_steps=self.virtual_timesteps)
            # print(f'Rate coded spike shape from step: {spikes.shape}', flush=True)  # Will this be [Port, Features, Virtual Timesteps]?
            current_spikes = spikes * mask
            
            buffer_idx = self.count % self.window_size
            self.spikes_buffer[buffer_idx] = current_spikes

            if (self.count + 1) % self.window_size == 0:
                self.update_scaling_factor(mask)

            self.count += 1

            return current_spikes
        else:
            if scaling_factor is None:
                raise Exception("Probabilities must be passed into 'encode' method for forward_sequences so the signal reconstruction is the same as in the rollout.")
            

            # Generate the spikes with the same probabilities that were used originally so the on-policy nature of PPO isn't broken.
            probs = cur_vals * scaling_factor * 1/self.desired_spike_rate
            spikes = spikegen.rate(probs, num_steps=self.virtual_timesteps)
            # print(f'Rate coded spike shape from sequence: {spikes.shape}', flush=True)
            spikes = spikes * mask
            
            assert not torch.isnan(spikes).any(), "NaN in spike input"
            assert not torch.isinf(spikes).any(), "Inf in spike input"

            return spikes



