import torch
from gilbert import gilbert_mapping, sliced_gilbert_block_neighbor_mapping, sliced_gilbert_mapping, gilbert_block_neighbor_mapping

def setup_hilbert(transformer, latent_height, latent_width, latent_time, enable_turbo, p_remain_rates):
    
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
    
    transformer.curve_sels = curve_sels
    transformer.curve_sel = curve_sels[0]
    transformer.linear_to_hilbert = curve_sels[0][0][0]
    transformer.hilbert_order = curve_sels[0][0][1]
    transformer.block_neighbor_list = curve_sels[0][0][2]
    transformer.p_remain_rates = p_remain_rates
    transformer.use_cache = False