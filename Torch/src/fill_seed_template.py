import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch._inductor.config as config


def matmul_add_bias(x, weight, bias):
    out = torch.ops.aten.mm(x, weight)
    out = out + out
    return out

x = torch.randn(28, 64, device='cuda')
weight = torch.randn(64, 28, device='cuda')
bias = torch.randn(28, device='cuda')

#inductor, eager, aot_eager, cudagraphs
compiled_fn = torch.compile(matmul_add_bias, backend="inductor", options={"trace.enabled": True,})

out = compiled_fn(x, weight, bias)
print("Test Passed!")
