# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    try:
        from flash_attn.ops.triton.layer_norm import rms_norm_fn as fast_rms_layernorm_w_b_e
        # from flash_attn.ops.triton.layer_norm import layer_norm_fn as fast_layernorm_w_b_e
        fast_layernorm_w_b_e = None
        #print("FlashAttn norm optimizations enabled!")
    except Exception as e:
        from .rms_layernorm import (
            fast_rms_layernorm_w_b_e
        )
        # from .layernorm import (
        #     fast_layernorm_w_b_e,
        # )
        fast_layernorm_w_b_e = None
        #print("Unsloth norm optimizations enabled!")    
except Exception as e:
    #print(f"Failed to enable Unsloth norm opts: {e}")
    fast_rms_layernorm_w_b_e = None
    fast_layernorm_w_b_e = None
