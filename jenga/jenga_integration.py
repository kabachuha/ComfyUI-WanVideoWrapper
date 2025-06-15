import torch
from gilbert import gilbert_mapping, sliced_gilbert_block_neighbor_mapping, sliced_gilbert_mapping, gilbert_block_neighbor_mapping

def setup_hilbert(cached_hilbert, latent_height, latent_width, latent_time, enable_turbo):
    
    key = f"{latent_height};{latent_width};{latent_time}"
    if key in cached_hilbert:
        return cached_hilbert[key]
    
    cached_hilbert.clear()
    
    # JULIAN: space curve related.
    res_rate_list = [0.75, 1.0] if enable_turbo else [1.0, 1.0]
    
    curve_sels = []
    for res_rate in res_rate_list:
        curve_sel = []
        latent_time_ = int(latent_time)
        latent_height_ = int(latent_height * res_rate)
        latent_width_ = int(latent_width * res_rate)
        LINEAR_TO_HILBERT, HILBERT_ORDER = sliced_gilbert_mapping(latent_time_, latent_height_, latent_width_)
        block_neighbor_list = sliced_gilbert_block_neighbor_mapping(latent_time_, latent_height_, latent_width_)
        
        # # linear settings.
        # LINEAR_TO_HILBERT = torch.arange(latent_time_ * latent_height_ * latent_width_, dtype=torch.long) # linear
        # HILBERT_ORDER = torch.arange(latent_time_ * latent_height_ * latent_width_, dtype=torch.long) # linear
        # block_neighbor_list = torch.zeros((math.ceil(latent_time_ * latent_height_ * latent_width_ / 128), math.ceil(latent_time_ * latent_height_ * latent_width_ / 128)), dtype=torch.bool)
        
        curve_sel.append([torch.tensor(LINEAR_TO_HILBERT, dtype=torch.long), torch.tensor(HILBERT_ORDER, dtype=torch.long), block_neighbor_list])
        curve_sels.append(curve_sel)
    
    cached_hilbert[key] = curve_sels
