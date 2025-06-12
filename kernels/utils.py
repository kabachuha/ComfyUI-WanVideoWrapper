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

import triton
MAX_FUSED_SIZE : int = 65536
next_power_of_2 = triton.next_power_of_2
import functools

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
import torch
torch_Tensor = torch.Tensor

# tl.math.tanh now is libdevice.tanh
from packaging.version import Version
import triton
import triton.language as tl
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh
    # No casting in old Triton versions
    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)
    pass
pass


def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass

if torch.cuda.device_count() > 1:
    torch_cuda_device = torch.cuda.device
else:
    from contextlib import nullcontext
    def torch_cuda_device(device): return nullcontext()
pass

