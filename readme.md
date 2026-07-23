# Fuzzing Deep Learning Compilers Through Systematic Exploration of Lowering Behaviors

> 🚀 A unified artifact for reproducing HARLO on **PyTorch Inductor** and **Apache TVM**.

## 🔍 Overview

HARLO is a harness-driven testing framework for exposing lowering-stage bugs in deep learning compilers.
This repository contains the implementation used in our paper, 
along with the scripts needed to reproduce the main experimental results.

### ✨ Highlights

- Supports **two DL compilers**: PyTorch Inductor and Apache TVM.
- Includes **Two reusable prompts** used by the LLM-assisted workflow.
- Summarizes the **84 previously unknown bugs** discovered by HARLO.
- Reproduces **coverage** results (Fig 7 and 8 in our paper)

## 📁 Repository Layout

- [`requirements.txt`](./requirements.txt) lists the Python dependencies used by the artifact.
- [`prompts/`](./prompts/) contains the prompt descriptions referenced in the paper.
- [`Torch/`](./Torch/) contains the PyTorch Inductor implementation, experiment scripts, and analysis utilities.
- [`TVM/`](./TVM/) contains the Apache TVM implementation, experiment scripts, and analysis utilities.


## 🛠️ Environment Setup

### 1. Create a Python environment (Python >=3.10)
```bash
pip install -r requirements.txt
```

### 2. Install Apache TVM from source
> Use the TVM source-build instructions from the official documentation:

- Official build guide: [TVM Install from Source](https://tvm.apache.org/docs/install/from_source.html)
- TVM commit `36f2502f` is the experiment version in our paper.

### 3. Install PyTorch Inductor from source
> Use the PyTorch source-build instructions from the official documentation:
- Official build guide: [PyTorch Build From Source](https://github.com/pytorch/pytorch#from-source)
- PyTorch commit `dac4f45f` is the experiment version in our paper

### 4. [Optional] Install `lcov` for coverage collection

- Official build guide: [lcov documentation](https://github.com/linux-test-project/lcov)
- After installing `lcov`, configure the executable path in the experiment code:
  + Update the `lcov_bin` setting in [`Torch/src/run_fuzz.py`](./Torch/src/run_fuzz.py) file to point to your local `lcov` executable.
  + Update the `lcov_bin` setting in [`TVM/src/fuzz.py`](./TVM/src/fuzz.py) file to point to your local `lcov` executable.


## 🔑 LLM Configuration
> HARLO uses LLM-backed generation utilities. The checked-in code no longer contains private tokens.
Please check the file [`Torch/src/utils/llm_utils.py`](./Torch/src/utils/llm_utils.py) and file [`TVM/src/llm_utils.py`](./TVM/src/llm_utils.py) for update

### Environment variables
- `LLM_API_KEY`: shared key used by the main generation/mutation client.
- `CODE_MODEL_API_KEY`: optional key for the code-oriented model client in the PyTorch pipeline.

If your deployment uses the same credential for both clients, you can simply set `CODE_MODEL_API_KEY` to the same value as `LLM_API_KEY`.


## ▶️ Running HARLO

### PyTorch Inductor
> Run the PyTorch pipeline from the `Torch` side of the repository using the scripts in [`Torch/`](./Torch/).
```bash
cd Torch/src
python run_fuzz.py 12h
```

### Apache TVM
> Run the TVM pipeline from the `TVM` side of the repository using the scripts in [`TVM/`](./TVM/).
```bash
cd TVM/src
python fuzz.py
```

## 🐛 Bug Inventory
> We uncovered **84 previously unknown bugs**: **39 in PyTorch Inductor** and **45 in Apache TVM**. Each issue number links directly to the corresponding public report.

**Status:** ✅ Fixed · 🔎 Confirmed · ⏳ Awaiting developer response

### PyTorch Inductor (39 bugs)

- ✅ **Fixed** · [Issue #174459](https://github.com/pytorch/pytorch/issues/174459) · `sort` · Incorrect code logic · Inconsistency
- ✅ **Fixed** · [Issue #174174](https://github.com/pytorch/pytorch/issues/174174) · `remainder` · Incorrect numerical computation · Inconsistency
- ✅ **Fixed** · [Issue #174172](https://github.com/pytorch/pytorch/issues/174172) · `copysign` · Type problem · Inconsistency
- 🔎 **Confirmed** · [Issue #174176](https://github.com/pytorch/pytorch/issues/174176) · `set` · Incorrect exception handling · Crash
- ✅ **Fixed** · [Issue #173987](https://github.com/pytorch/pytorch/issues/173987) · `eq` · Type problem · Inconsistency
- ✅ **Fixed** · [Issue #173624](https://github.com/pytorch/pytorch/issues/173624) · `vdot + matrix_exp_backward` · Incorrect exception handling · Segmentation fault
- ✅ **Fixed** · [Issue #173478](https://github.com/pytorch/pytorch/issues/173478) · `atan + special_psi` · Incorrect code logic · Inconsistency
- ⏳ **Awaiting** · [Issue #173476](https://github.com/pytorch/pytorch/issues/173476) · `lstm` · Unclassified · Crash/hang
- ✅ **Fixed** · [Issue #173307](https://github.com/pytorch/pytorch/issues/173307) · `infinitely_differentiable_gelu_backward` · Type problem · Inconsistency
- 🔎 **Confirmed** · [Issue #173054](https://github.com/pytorch/pytorch/issues/173054) · `hardswish_backward + add` · Incorrect code logic · Inconsistency
- ✅ **Fixed** · [Issue #173052](https://github.com/pytorch/pytorch/issues/173052) · `linalg_pinv` · Tensor-shape problem · Inconsistency
- 🔎 **Confirmed** · [Issue #173046](https://github.com/pytorch/pytorch/issues/173046) · `rnn_tanh` · Tensor-shape problem · Segmentation fault
- ✅ **Fixed** · [Issue #172155](https://github.com/pytorch/pytorch/issues/172155) · `conv3d` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #172073](https://github.com/pytorch/pytorch/issues/172073) · [PR #172171](https://github.com/pytorch/pytorch/pull/172171) · `scalar_tensor` · Tensor-shape problem · Crash
- ✅ **Fixed** · [Issue #172068](https://github.com/pytorch/pytorch/issues/172068) · `ldexp` · Incorrect numerical computation · Inconsistency
- ✅ **Fixed** · [Issue #172066](https://github.com/pytorch/pytorch/issues/172066) · `rnn_relu` · Incorrect exception handling · Segmentation fault
- ✅ **Fixed** · [Issue #172061](https://github.com/pytorch/pytorch/issues/172061) · `affine_grid_generator` · Incorrect code logic · Inconsistency
- ✅ **Fixed** · [Issue #171375](https://github.com/pytorch/pytorch/issues/171375) · `lift` · Type problem · Crash
- ✅ **Fixed** · [Issue #171370](https://github.com/pytorch/pytorch/issues/171370) · `unfold_backward` · Tensor-shape problem · Crash
- ✅ **Fixed** · [Issue #171366](https://github.com/pytorch/pytorch/issues/171366) · `view_copy` · Tensor-shape problem · Crash
- ✅ **Fixed** · [Issue #171284](https://github.com/pytorch/pytorch/issues/171284) · `gru` · Incorrect exception handling · Segmentation fault
- 🔎 **Confirmed** · [Issue #171283](https://github.com/pytorch/pytorch/issues/171283) · `uniform` · Incorrect assignment · Crash
- 🔎 **Confirmed** · [Issue #171283](https://github.com/pytorch/pytorch/issues/171283) · `random` · Incorrect assignment · Crash
- ✅ **Fixed** · [Issue #171282](https://github.com/pytorch/pytorch/issues/171282) · `binary_cross_entropy_with_logits` · Type problem · Inconsistency
- ✅ **Fixed** · [Issue #171282](https://github.com/pytorch/pytorch/issues/171282) · `cumprod_backward` · Type problem · Inconsistency
- ✅ **Fixed** · [Issue #171262](https://github.com/pytorch/pytorch/issues/171262) · `isin` · Tensor-shape problem · Inconsistency
- 🔎 **Confirmed** · [Issue #171259](https://github.com/pytorch/pytorch/issues/171259) · `multilabel_margin_loss` · Incorrect exception handling · Crash
- ✅ **Fixed** · [Issue #171256](https://github.com/pytorch/pytorch/issues/171256) · `trapezoid` · Incorrect code logic · Inconsistency
- 🔎 **Confirmed** · [Issue #170782](https://github.com/pytorch/pytorch/issues/170782) · `_no_grad_embedding_renorm_` · Concurrency · Crash
- 🔎 **Confirmed** · [Issue #170762](https://github.com/pytorch/pytorch/issues/170762) · `cudnn_batch_norm_backward` · Type problem · Inconsistency
- 🔎 **Confirmed** · [Issue #170745](https://github.com/pytorch/pytorch/issues/170745) · `poisson_nll_loss` · Type problem · Inconsistency
- 🔎 **Confirmed** · [Issue #170688](https://github.com/pytorch/pytorch/issues/170688) · `__interpolate` · Incorrect code logic · Inconsistency
- 🔎 **Confirmed** · [Issue #170686](https://github.com/pytorch/pytorch/issues/170686) · `_to_sparse_bsc` · API misuse · Crash
- 🔎 **Confirmed** · [Issue #170684](https://github.com/pytorch/pytorch/issues/170684) · `cosine_similarity` · Type problem · Inconsistency
- ✅ **Fixed** · [Issue #170682](https://github.com/pytorch/pytorch/issues/170682) · `__upsample_nearest` · Incorrect assignment · Inconsistency
- 🔎 **Confirmed** · [Issue #170642](https://github.com/pytorch/pytorch/issues/170642) · `_cudnn_init_dropout_state` · Type problem · Crash
- ✅ **Fixed** · [Issue #170520](https://github.com/pytorch/pytorch/issues/170520) · `_fused_dropout` · Incorrect numerical computation · Crash
- ✅ **Fixed** · [Issue #169939](https://github.com/pytorch/pytorch/issues/169939) · CUDAGraph options · Incorrect exception handling · Crash
- ✅ **Fixed** · [Issue #162473](https://github.com/pytorch/pytorch/issues/162473) · `nonzero_static` · Incorrect code logic · Segmentation fault

### Apache TVM (45 bugs)

- ⏳ **Awaiting** · [Issue #18380](https://github.com/apache/tvm/issues/18380) · `AutoInline` · Unclassified · Crash
- ✅ **Fixed** · [Issue #18769](https://github.com/apache/tvm/issues/18769) · `where + tan` · Type problem · Crash
- ✅ **Fixed** · [Issue #18767](https://github.com/apache/tvm/issues/18767) · `fast_tanh` · Type problem · Crash
- ✅ **Fixed** · [Issue #18767](https://github.com/apache/tvm/issues/18767) · `fast_exp` · Type problem · Crash
- ✅ **Fixed** · [Issue #18765](https://github.com/apache/tvm/issues/18765) · `get_const_tuple` · Type problem · Crash/hang
- ✅ **Fixed** · [Issue #18729](https://github.com/apache/tvm/issues/18729) · `acos` · Type problem · Crash
- ✅ **Fixed** · [Issue #18729](https://github.com/apache/tvm/issues/18729) · `asin` · Type problem · Crash
- ✅ **Fixed** · [Issue #18425](https://github.com/apache/tvm/issues/18425) · `llvm.log2` · Type problem · Crash
- ✅ **Fixed** · [Issue #18425](https://github.com/apache/tvm/issues/18425) · `llvm.log` · Type problem · Crash
- ⏳ **Awaiting** · [Issue #18424](https://github.com/apache/tvm/issues/18424) · `RewriteParallelVectorizeUnroll` · Unclassified · Segmentation fault
- ⏳ **Awaiting** · [Issue #18423](https://github.com/apache/tvm/issues/18423) · `RewriteFuseSplitParallelVectorize` · Unclassified · Segmentation fault
- ⏳ **Awaiting** · [Issue #18393](https://github.com/apache/tvm/issues/18393) · `FlattenBuffer` · Unclassified · Segmentation fault
- ✅ **Fixed** · [Issue #18387](https://github.com/apache/tvm/issues/18387) · `tir.AssertStmt` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #18387](https://github.com/apache/tvm/issues/18387) · `tir.StringImm` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #18383](https://github.com/apache/tvm/issues/18383) · TIR scheduling · Incorrect code logic · Crash
- ✅ **Fixed** · [Issue #18381](https://github.com/apache/tvm/issues/18381) · `exp` · Type problem · Crash
- ✅ **Fixed** · [Issue #18334](https://github.com/apache/tvm/issues/18334) · `FP8ComputeLegalize` · Typo · Crash
- ⏳ **Awaiting** · [Issue #17997](https://github.com/apache/tvm/issues/17997) · `conv2d_hwcn` · Unclassified · Inconsistency
- ✅ **Fixed** · [Issue #17996](https://github.com/apache/tvm/issues/17996) · `llvm.log10` · Type problem · Crash
- ✅ **Fixed** · [Issue #17995](https://github.com/apache/tvm/issues/17995) · `take` · Tensor-shape problem · Crash
- ✅ **Fixed** · [Issue #17976](https://github.com/apache/tvm/issues/17976) · `erf` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17974](https://github.com/apache/tvm/issues/17974) · `topk` · Type problem · Crash
- ✅ **Fixed** · [Issue #17969](https://github.com/apache/tvm/issues/17969) · `asinh` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17969](https://github.com/apache/tvm/issues/17969) · `acosh` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17969](https://github.com/apache/tvm/issues/17969) · `atanh` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17961](https://github.com/apache/tvm/issues/17961) · `searchsorted` scheduling · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17953](https://github.com/apache/tvm/issues/17953) · `sqrt / rsqrt` · Type problem · Crash
- ⏳ **Awaiting** · [Issue #17949](https://github.com/apache/tvm/issues/17949) · `tensordot` · Unclassified · Inconsistency
- ✅ **Fixed** · [Issue #17947](https://github.com/apache/tvm/issues/17947) · `expand_like` · Incorrect code logic · Inconsistency
- ✅ **Fixed** · [Issue #17945](https://github.com/apache/tvm/issues/17945) · `asin` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17945](https://github.com/apache/tvm/issues/17945) · `acos` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17945](https://github.com/apache/tvm/issues/17945) · `atan` · Missing regression coverage · Crash
- ✅ **Fixed** · [Issue #17941](https://github.com/apache/tvm/issues/17941) · `conv1d` · Incorrect code logic · Crash
- ✅ **Fixed** · [Issue #17938](https://github.com/apache/tvm/issues/17938) · `global_pool` · Type problem · Crash
- ⏳ **Awaiting** · [Issue #17937](https://github.com/apache/tvm/issues/17937) · `group_norm` · Unclassified · Crash
- ⏳ **Awaiting** · [Issue #17936](https://github.com/apache/tvm/issues/17936) · `group_norm` · Unclassified · Crash
- ⏳ **Awaiting** · [Issue #17935](https://github.com/apache/tvm/issues/17935) · `log_softmax` · Unclassified · Inconsistency
- ⏳ **Awaiting** · [Issue #17934](https://github.com/apache/tvm/issues/17934) · `pool_grad` · Unclassified · Crash
- ✅ **Fixed** · [Issue #17933](https://github.com/apache/tvm/issues/17933) · `scale_shift_nhwc` · Incorrect code logic · Crash
- ⏳ **Awaiting** · [Issue #17870](https://github.com/apache/tvm/issues/17870) · `MakePackedAPI` · Unclassified · Crash
- ✅ **Fixed** · [Issue #17848](https://github.com/apache/tvm/issues/17848) · `batch_to_space_nd` · Typo · Crash
- ✅ **Fixed** · [Issue #17848](https://github.com/apache/tvm/issues/17848) · `bitpack` · Typo · Crash
- ✅ **Fixed** · [Issue #17847](https://github.com/apache/tvm/issues/17847) · `topi.nn.simplify` · Incorrect code logic · Crash
- ✅ **Fixed** · [Issue #17845](https://github.com/apache/tvm/issues/17845) · `upsampling` · Typo · Crash
- ✅ **Fixed** · [Issue #17844](https://github.com/apache/tvm/issues/17844) · `softmax` · Typo · Crash



## 📈 Reproducing Coverage Results

1. Configure your local compiler build paths and `lcov_bin`.
2. Run the coverage script under the corresponding `experiments` directory.
3. Inspect the generated raw results in that experiment output directory.
4. Run the plotting script in the same compiler-specific experiment folder to regenerate the paper figures.

### Plotting coverage curves
> Both compiler directories include plotting scripts for turning the collected coverage data into figures:

- PyTorch plotting scripts: [`Torch/experiments/`](./Torch/experiments/)
- TVM plotting scripts: [`TVM/experiments/`](./TVM/experiments/)
