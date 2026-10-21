# Configuration-Driven Fuzzing for Deep Learning Compiler Lowering

> 🚀 A unified artifact for reproducing LPTGen on **PyTorch Inductor** and **Apache TVM**.

## 🔍 Overview

LPTGen is a harness-driven testing framework for exposing lowering-stage bugs in deep learning compilers.
This repository contains the implementation used in our paper, 
along with the scripts needed to reproduce the main experimental results.

### ✨ Highlights

- Supports **two DL compilers**: PyTorch Inductor and Apache TVM.
- Includes **Two reusable prompts** used by the LLM-assisted workflow.
- Summarizes the **84 previously unknown bugs** discovered by LPTGen.
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
> LPTGen uses LLM-backed generation utilities. The checked-in code no longer contains private tokens.
Please check the file [`Torch/src/utils/llm_utils.py`](./Torch/src/utils/llm_utils.py) and file [`TVM/src/llm_utils.py`](./TVM/src/llm_utils.py) for update

### Environment variables
- `LLM_API_KEY`: shared key used by the main generation/mutation client.
- `CODE_MODEL_API_KEY`: optional key for the code-oriented model client in the PyTorch pipeline.

If your deployment uses the same credential for both clients, you can simply set `CODE_MODEL_API_KEY` to the same value as `LLM_API_KEY`.


## ▶️ Running LPTGen

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

- ✅ **Fixed** · [Issue #174459：inductor inconsistent with eager for aten.sort(stable=None) on CUDA](https://github.com/pytorch/pytorch/issues/174459)
- ✅ **Fixed** · [Issue #174174：inductor produces inconsistent results with inductor for aten.remainder](https://github.com/pytorch/pytorch/issues/174174)
- ✅ **Fixed** · [Issue #174172：TorchInductor produce differnt results with eager about the copysign operator](https://github.com/pytorch/pytorch/issues/174172)
- ✅ **Fixed** · [Issue #174176：Inductor crashes with INTERNAL ASSERT in FunctionalizeFallbackKernel](https://github.com/pytorch/pytorch/issues/174176)
- ✅ **Fixed** · [Issue #173987：Inductor produces different results from eager for aten.eq with BF16 tensor and FP32 scalar](https://github.com/pytorch/pytorch/issues/173987)
- ✅ **Fixed** · [Issue #173624：SegFault in matrix_exp_backward with Scalar Input](https://github.com/pytorch/pytorch/issues/173624)
- ✅ **Fixed** · [Issue #173478：Numerical Discrepancy Between Eager and Inductor Backends for atan + special_psi Composition](https://github.com/pytorch/pytorch/issues/173478)
- ⏳ **Awaiting** · [Issue #173476：Unrecoverable CUDA deadlock in torch.ops.aten.lstm - process becomes unresponsive](https://github.com/pytorch/pytorch/issues/173476)
- ✅ **Fixed** · [Issue #173307：Numerical divergence between eager and inductor for infinitely_differentiable_gelu_backward in BF16](https://github.com/pytorch/pytorch/issues/173307)
- 🔎 **Confirmed** · [Issue #173054：TorchInductor produces incorrect result for aten.hardswish_backward followed by aten.add](https://github.com/pytorch/pytorch/issues/173054)
- ✅ **Fixed** · [Issue #173052：aten.linalg_pinv returns inconsistent output shape between eager and inductor](https://github.com/pytorch/pytorch/issues/173052)
- 🔎 **Confirmed** · [Issue #173046：compilion with eager for the rnn_tanh operator leads to a segfault](https://github.com/pytorch/pytorch/issues/173046)
- ✅ **Fixed** · [Issue #172155：Inductor crashes with NoValidChoicesError on valid conv3d](https://github.com/pytorch/pytorch/issues/172155)
- ✅ **Fixed** · [Issue #172073：Inductor backend crashes when adding 0-dim Complex tensor](https://github.com/pytorch/pytorch/issues/172073) · [PR #172171](https://github.com/pytorch/pytorch/pull/172171)
- ✅ **Fixed** · [Issue #172068：ldexp float16 backend inconsistency on CUDA](https://github.com/pytorch/pytorch/issues/172068)
- ✅ **Fixed** · [Issue #172066：when using torch.ops.aten.rnn_relu on CUDA with torch.compile](https://github.com/pytorch/pytorch/issues/172066)
- ✅ **Fixed** · [Issue #172061：Inductor produces large numerical errors for affine_grid_generator on CUDA with fp16/bf16](https://github.com/pytorch/pytorch/issues/172061)
- ✅ **Fixed** · [Issue #171375：aten.lift triggers internal assert failure in Inductor functionalization pass](https://github.com/pytorch/pytorch/issues/171375)
- ✅ **Fixed** · [Issue #171370：Inductor fails on aten.unfold_backward while eager backend works](https://github.com/pytorch/pytorch/issues/171370)
- ✅ **Fixed** · [Issue #171366：aten.view_copy produces inconsistent output shapes between eager and inductor](https://github.com/pytorch/pytorch/issues/171366)
- ✅ **Fixed** · [Issue #171284：GRU: Segmentation fault (core dumped)](https://github.com/pytorch/pytorch/issues/171284)
- 🔎 **Confirmed** · [Issue #171283：aten.uniform cannot be called with keyword argument from](https://github.com/pytorch/pytorch/issues/171283)
- 🔎 **Confirmed** · [Issue #171283：aten.uniform cannot be called with keyword argument from](https://github.com/pytorch/pytorch/issues/171283)
- ✅ **Fixed** · [Issue #171282：binary_cross_entropy_with_logits returns different dtypes between Eager and Inductor](https://github.com/pytorch/pytorch/issues/171282)
- ✅ **Fixed** · [Issue #171282：binary_cross_entropy_with_logits returns different dtypes between Eager and Inductor](https://github.com/pytorch/pytorch/issues/171282)
- ✅ **Fixed** · [Issue #171262：TorchInductor returns wrong output shape for aten.isin with scalar input](https://github.com/pytorch/pytorch/issues/171262)
- 🔎 **Confirmed** · [Issue #171259：torch.AcceleratorError: CUDA error: device-side assert triggered](https://github.com/pytorch/pytorch/issues/171259)
- ✅ **Fixed** · [Issue #171256：Significant numerically inconsistent results for aten.trapezoid](https://github.com/pytorch/pytorch/issues/171256)
- ✅ **Fixed** · [Issue #170782：TorchInductor crashes with CUDA illegal memory access for aten._no_grad_embedding_renorm_](https://github.com/pytorch/pytorch/issues/170782)
- 🔎 **Confirmed** · [Issue #170762：aten.cudnn_batch_norm_backward produces numerical mismatch between eager and inductor](https://github.com/pytorch/pytorch/issues/170762)
- 🔎 **Confirmed** · [Issue #170745：poisson_nll_loss output differnt results between eager and inductor](https://github.com/pytorch/pytorch/issues/170745)
- ✅ **Fixed** · [Issue #170688：Numerical mismatch between eager and inductor for aten.__interpolate](https://github.com/pytorch/pytorch/issues/170688)
- 🔎 **Confirmed** · [Issue #170686：Inductor fails on aten._to_sparse_bsc due to incorrect stride assumption for Sparse BSC tensors](https://github.com/pytorch/pytorch/issues/170686)
- 🔎 **Confirmed** · [Issue #170684：Numerical inconsistency with eager for aten.cosine_similarity](https://github.com/pytorch/pytorch/issues/170684)
- ✅ **Fixed** · [Issue #170682：aten.__upsample_nearest gives different results on Inductor vs Eager with bfloat16](https://github.com/pytorch/pytorch/issues/170682)
- 🔎 **Confirmed** · [Issue #170642：_cudnn_init_dropout_state crashes with dtype != Byte](https://github.com/pytorch/pytorch/issues/170642)
- ✅ **Fixed** · [Issue #170520：aten._fused_dropout lacks guard for p=0, causing division-by-zero](https://github.com/pytorch/pytorch/issues/170520)
- ✅ **Fixed** · [Issue #169939：TypeError in CudagraphsBackend when using torch.compile with options parameter](https://github.com/pytorch/pytorch/issues/169939)
- ✅ **Fixed** · [Issue #162473：Floating point exception (core dumped) for nonzero_static operator](https://github.com/pytorch/pytorch/issues/162473)

### Apache TVM (45 bugs)

- ✅ **Fixed** · [Issue #18380：tvm.tir.schedule.schedule.ScheduleError](https://github.com/apache/tvm/issues/18380)
- ✅ **Fixed** · [Issue #18769：Cannot find intrinsic declaration, possible type mismatch: llvm.sin](https://github.com/apache/tvm/issues/18769)
- ✅ **Fixed** · [Issue #18767：Code generation fails when compiling topi.fast_tanh for int16 dtype with LLVM target](https://github.com/apache/tvm/issues/18767)
- ✅ **Fixed** · [Issue #18767：Code generation fails when compiling topi.fast_tanh for int16 dtype with LLVM target](https://github.com/apache/tvm/issues/18767)
- ✅ **Fixed** · [Issue #18765：topi.get_const_tuple hangs indefinitely when passed a te.Placeholder](https://github.com/apache/tvm/issues/18765)
- ✅ **Fixed** · [Issue #18729：topi.acos accepts integer input and crashes in LowerIntrin with InternalError](https://github.com/apache/tvm/issues/18729)
- ✅ **Fixed** · [Issue #18729：topi.acos accepts integer input and crashes in LowerIntrin with InternalError](https://github.com/apache/tvm/issues/18729)
- ✅ **Fixed** · [Issue #18425：Cannot find intrinsic declaration, possible type mismatch: llvm.log2](https://github.com/apache/tvm/issues/18425)
- ✅ **Fixed** · [Issue #18425：Cannot find intrinsic declaration, possible type mismatch: llvm.log2](https://github.com/apache/tvm/issues/18425)
- ✅ **Fixed** · [Issue #18424：Segfault when applying Parallel during TIR schedule rewriting](https://github.com/apache/tvm/issues/18424)
- ✅ **Fixed** · [Issue #18423：MetaSchedule tune_tir crashes with ScheduleError in RewriteFuseSplitParallelVectorize](https://github.com/apache/tvm/issues/18423)
- ✅ **Fixed** · [Issue #18393：Segfault in TVM when building TIR module with pragma_unroll_explicit annotations](https://github.com/apache/tvm/issues/18393)
- ✅ **Fixed** · [Issue #18387：Do not have a default for tir.AssertStmt](https://github.com/apache/tvm/issues/18387)
- ✅ **Fixed** · [Issue #18387：Do not have a default for tir.AssertStmt](https://github.com/apache/tvm/issues/18387)
- ✅ **Fixed** · [Issue #18383：Crash when multiple PrimFunc objects are present in IRModule](https://github.com/apache/tvm/issues/18383)
- ✅ **Fixed** · [Issue #18381：Cannot find intrinsic declaration, possible type mismatch: llvm.exp](https://github.com/apache/tvm/issues/18381)
- ✅ **Fixed** · [Issue #18334：Fix conflict parameter name promote_dtye in FP8ComputeLegalize](https://github.com/apache/tvm/issues/18334)
- ⏳ **Awaiting** · [Issue #17997：Inconcistency after meta schedule for the conv2d_hwcn operator](https://github.com/apache/tvm/issues/17997)
- ✅ **Fixed** · [Issue #17996：Cannot find intrinsic declaration, possible type mismatch: llvm.log10](https://github.com/apache/tvm/issues/17996)
- ✅ **Fixed** · [Issue #17995：Fix zero-extent loops in PerStoreFeature to prevent crashes](https://github.com/apache/tvm/issues/17995)
- ✅ **Fixed** · [Issue #17976：Add registion for the operator erf in llvm](https://github.com/apache/tvm/issues/17976)
- ✅ **Fixed** · [Issue #17974：Fix FLOP estimation for EvaluateNode](https://github.com/apache/tvm/issues/17974)
- ✅ **Fixed** · [Issue #17969：Add registion for the operator asinh, acosh, atanh in llvm](https://github.com/apache/tvm/issues/17969)
- ✅ **Fixed** · [Issue #17969：Add registion for the operator asinh, acosh, atanh in llvm](https://github.com/apache/tvm/issues/17969)
- ✅ **Fixed** · [Issue #17969：Add registion for the operator asinh, acosh, atanh in llvm](https://github.com/apache/tvm/issues/17969)
- ✅ **Fixed** · [Issue #17961：Fix basic FLOP estimation for WhileNode](https://github.com/apache/tvm/issues/17961)
- ✅ **Fixed** · [Issue #17953：Fix sqrt/rsqrt Compatibility with Integer Data Types](https://github.com/apache/tvm/issues/17953)
- ⏳ **Awaiting** · [Issue #17949：tensordot produce inconsistency inference results when execuing twice under the same inputs](https://github.com/apache/tvm/issues/17949)
- ✅ **Fixed** · [Issue #17947：Meta schedule produce inconsistant inference results](https://github.com/apache/tvm/issues/17947)
- ✅ **Fixed** · [Issue #17945：Add registion for the operator asin and acos in llvm](https://github.com/apache/tvm/issues/17945)
- ✅ **Fixed** · [Issue #17945：Add registion for the operator asin and acos in llvm](https://github.com/apache/tvm/issues/17945)
- ✅ **Fixed** · [Issue #17945：Add registion for the operator asin and acos in llvm](https://github.com/apache/tvm/issues/17945)
- ✅ **Fixed** · [Issue #17941：tvm._ffi.base.TVMError: std: :bad_alloc](https://github.com/apache/tvm/issues/17941)
- ✅ **Fixed** · [Issue #17938：InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero](https://github.com/apache/tvm/issues/17938)
- ⏳ **Awaiting** · [Issue #17937：InternalError: Check failed: (ret_ex.dtype() == var.dtype()) is false](https://github.com/apache/tvm/issues/17937)
- ✅ **Fixed** · [Issue #17936：InternalError: Check failed: (eval_vec_) is false](https://github.com/apache/tvm/issues/17936)
- ⏳ **Awaiting** · [Issue #17935：MetaSchedule lead to a significant difference in the reference results for the log_softmax operator](https://github.com/apache/tvm/issues/17935)
- ⏳ **Awaiting** · [Issue #17934：parallel_for_dynamic error with ScheduleError: (not rendered)](https://github.com/apache/tvm/issues/17934)
- ✅ **Fixed** · [Issue #17933：sch.mod returns None after meta scheduling](https://github.com/apache/tvm/issues/17933)
- ⏳ **Awaiting** · [Issue #17870：Incorrect transform for the MakePackedAPI pass](https://github.com/apache/tvm/issues/17870)
- ✅ **Fixed** · [Issue #17848：Fix docstring in batch_to_space_nd and bitpack](https://github.com/apache/tvm/issues/17848)
- ✅ **Fixed** · [Issue #17848：Fix docstring in batch_to_space_nd and bitpack](https://github.com/apache/tvm/issues/17848)
- ✅ **Fixed** · [Issue #17847：Array declaration of expr occurred multiple times](https://github.com/apache/tvm/issues/17847)
- ✅ **Fixed** · [Issue #17845：fixing incorrect docstring in upsampling.py](https://github.com/apache/tvm/issues/17845)
- ✅ **Fixed** · [Issue #17844：Fix incorrect docstring in topi softmax ](https://github.com/apache/tvm/issues/17844)



## 📈 Reproducing Coverage Results

1. Configure your local compiler build paths and `lcov_bin`.
2. Run the coverage script under the corresponding `experiments` directory.
3. Inspect the generated raw results in that experiment output directory.
4. Run the plotting script in the same compiler-specific experiment folder to regenerate the paper figures.

### Plotting coverage curves
> Both compiler directories include plotting scripts for turning the collected coverage data into figures:

- PyTorch plotting scripts: [`Torch/experiments/`](./Torch/experiments/)
- TVM plotting scripts: [`TVM/experiments/`](./TVM/experiments/)
