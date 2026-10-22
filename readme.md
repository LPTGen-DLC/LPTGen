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

## 🐛 Bugs Detected by LPTGen (Fixed)

> We uncovered **79 previously unknown bugs**: **40 in PyTorch Inductor** and **39 in Apache TVM**. 
Of these, **64 have been fixed**: **29 in PyTorch Inductor** and **35 in Apache TVM**.

The list below includes LPTGen detected bugs that has been fixed by developers. 
Each item links directly to the corresponding public issue report. 
`Fixed *N` indicates that the linked issue covers N bugs.

### PyTorch Inductor (29 fixed bugs)

- ✅ **Fixed** · [inductor inconsistent with eager for aten.sort(stable=None) on CUDA](https://github.com/pytorch/pytorch/issues/174459)
- ✅ **Fixed** · [TorchInductor produce differnt results with eager about the copysign operator](https://github.com/pytorch/pytorch/issues/174172)
- ✅ **Fixed** · [Inductor crashes with INTERNAL ASSERT in FunctionalizeFallbackKernel](https://github.com/pytorch/pytorch/issues/174176)
- ✅ **Fixed** · [Inductor produces different results from eager for aten.eq with BF16 tensor and FP32 scalar](https://github.com/pytorch/pytorch/issues/173987)
- ✅ **Fixed** · [SegFault in matrix_exp_backward with Scalar Input](https://github.com/pytorch/pytorch/issues/173624)
- ✅ **Fixed** · [Numerical Discrepancy Between Eager and Inductor Backends for atan + special_psi Composition](https://github.com/pytorch/pytorch/issues/173478)
- ✅ **Fixed** · [Numerical divergence between eager and inductor for infinitely_differentiable_gelu_backward in BF16](https://github.com/pytorch/pytorch/issues/173307)
- ✅ **Fixed** · [aten.linalg_pinv returns inconsistent output shape between eager and inductor](https://github.com/pytorch/pytorch/issues/173052)
- ✅ **Fixed** · [Numerical mismatch between eager and inductor for aten.__interpolate](https://github.com/pytorch/pytorch/issues/170688)
- ✅ **Fixed** · [Inductor crashes with NoValidChoicesError on valid conv3d](https://github.com/pytorch/pytorch/issues/172155)
- ✅ **Fixed** · [Inductor backend crashes when adding 0-dim Complex tensor](https://github.com/pytorch/pytorch/issues/172073) · [PR #172171](https://github.com/pytorch/pytorch/pull/172171)
- ✅ **Fixed** · [ldexp float16 backend inconsistency on CUDA](https://github.com/pytorch/pytorch/issues/172068)
- ✅ **Fixed** · [when using torch.ops.aten.rnn_relu on CUDA with torch.compile](https://github.com/pytorch/pytorch/issues/172066)
- ✅ **Fixed** · [Inductor produces large numerical errors for affine_grid_generator on CUDA with fp16/bf16](https://github.com/pytorch/pytorch/issues/172061)
- ✅ **Fixed** · [aten.lift triggers internal assert failure in Inductor functionalization pass](https://github.com/pytorch/pytorch/issues/171375)
- ✅ **Fixed** · [Inductor fails on aten.unfold_backward while eager backend works](https://github.com/pytorch/pytorch/issues/171370)
- ✅ **Fixed** · [aten.view_copy produces inconsistent output shapes between eager and inductor](https://github.com/pytorch/pytorch/issues/171366)
- ✅ **Fixed** · [GRU: Segmentation fault (core dumped)](https://github.com/pytorch/pytorch/issues/171284)
- ✅ **Fixed** *2 · [binary_cross_entropy_with_logits returns different dtypes between Eager and Inductor](https://github.com/pytorch/pytorch/issues/171282)
- ✅ **Fixed** · [TorchInductor returns wrong output shape for aten.isin with scalar input](https://github.com/pytorch/pytorch/issues/171262)
- ✅ **Fixed** · [Significant numerically inconsistent results for aten.trapezoid](https://github.com/pytorch/pytorch/issues/171256)
- ✅ **Fixed** · [TorchInductor crashes with CUDA illegal memory access for aten._no_grad_embedding_renorm_](https://github.com/pytorch/pytorch/issues/170782)
- ✅ **Fixed** · [aten.__upsample_nearest gives different results on Inductor vs Eager with bfloat16](https://github.com/pytorch/pytorch/issues/170682)
- ✅ **Fixed** · [aten._fused_dropout lacks guard for p=0, causing division-by-zero](https://github.com/pytorch/pytorch/issues/170520)
- ✅ **Fixed** · [inductor produces inconsistent results with inductor for aten.remainder](https://github.com/pytorch/pytorch/issues/174174)
- ✅ **Fixed** · [TypeError in CudagraphsBackend when using torch.compile with options parameter](https://github.com/pytorch/pytorch/issues/169939)
- ✅ **Fixed** · [Floating point exception (core dumped) for nonzero_static operator](https://github.com/pytorch/pytorch/issues/162473)
- ✅ **Fixed** · [inductor backend produces numerically divergent results vs eager for aten.cumprod_backward](https://github.com/pytorch/pytorch/issues/172062)

### Apache TVM (35 fixed bugs)

- ✅ **Fixed** · [tvm.tir.schedule.schedule.ScheduleError](https://github.com/apache/tvm/issues/18380)
- ✅ **Fixed** · [Cannot find intrinsic declaration, possible type mismatch: llvm.sin](https://github.com/apache/tvm/issues/18769)
- ✅ **Fixed** *2 · [Code generation fails when compiling topi.fast_tanh for int16 dtype with LLVM target](https://github.com/apache/tvm/issues/18767)
- ✅ **Fixed** · [topi.get_const_tuple hangs indefinitely when passed a te.Placeholder](https://github.com/apache/tvm/issues/18765)
- ✅ **Fixed** *2 · [topi.acos accepts integer input and crashes in LowerIntrin with InternalError](https://github.com/apache/tvm/issues/18729)
- ✅ **Fixed** · [Meta schedule produce inconsistant inference results](https://github.com/apache/tvm/issues/17947)
- ✅ **Fixed** *2 · [Cannot find intrinsic declaration, possible type mismatch: llvm.log2](https://github.com/apache/tvm/issues/18425)
- ✅ **Fixed** · [Segfault when applying Parallel during TIR schedule rewriting](https://github.com/apache/tvm/issues/18424)
- ✅ **Fixed** · [MetaSchedule tune_tir crashes with ScheduleError in RewriteFuseSplitParallelVectorize](https://github.com/apache/tvm/issues/18423)
- ✅ **Fixed** · [Segfault in TVM when building TIR module with pragma_unroll_explicit annotations](https://github.com/apache/tvm/issues/18393)
- ✅ **Fixed** *2 · [Do not have a default for tir.AssertStmt](https://github.com/apache/tvm/issues/18387)
- ✅ **Fixed** · [Crash when multiple PrimFunc objects are present in IRModule](https://github.com/apache/tvm/issues/18383)
- ✅ **Fixed** · [Cannot find intrinsic declaration, possible type mismatch: llvm.exp](https://github.com/apache/tvm/issues/18381)
- ✅ **Fixed** · [Cannot find intrinsic declaration, possible type mismatch: llvm.log10](https://github.com/apache/tvm/issues/17996)
- ✅ **Fixed** · [Fix zero-extent loops in PerStoreFeature to prevent crashes](https://github.com/apache/tvm/issues/17995)
- ✅ **Fixed** · [Add registion for the operator erf in llvm](https://github.com/apache/tvm/issues/17976)
- ✅ **Fixed** · [Fix FLOP estimation for EvaluateNode](https://github.com/apache/tvm/issues/17974)
- ✅ **Fixed** *3 · [Add registion for the operator asinh, acosh, atanh in llvm](https://github.com/apache/tvm/issues/17969)
- ✅ **Fixed** · [Fix basic FLOP estimation for WhileNode](https://github.com/apache/tvm/issues/17961)
- ✅ **Fixed** · [Fix sqrt/rsqrt Compatibility with Integer Data Types](https://github.com/apache/tvm/issues/17953)
- ✅ **Fixed** *3 · [Add registion for the operator asin and acos in llvm](https://github.com/apache/tvm/issues/17945)
- ✅ **Fixed** · [tvm._ffi.base.TVMError: std: :bad_alloc](https://github.com/apache/tvm/issues/17941)
- ✅ **Fixed** · [InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero](https://github.com/apache/tvm/issues/17938)
- ✅ **Fixed** · [InternalError: Check failed: (eval_vec_) is false](https://github.com/apache/tvm/issues/17936)
- ✅ **Fixed** · [parallel_for_dynamic error with ScheduleError: (not rendered)](https://github.com/apache/tvm/issues/17934)
- ✅ **Fixed** · [sch.mod returns None after meta scheduling](https://github.com/apache/tvm/issues/17933)
- ✅ **Fixed** · [Array declaration of expr occurred multiple times](https://github.com/apache/tvm/issues/17847)


In addition, LPTGen uncovered five documentation bugs during operator schema extraction and configuration instantiation. 
All five were fixed by developers and are excluded from the compiler bug counts reported above.

- ✅ **Fixed** · [Fix conflict parameter name promote_dtye in FP8ComputeLegalize](https://github.com/apache/tvm/issues/18334)
- ✅ **Fixed** · [Fix incorrect docstring in topi softmax ](https://github.com/apache/tvm/issues/17844)
- ✅ **Fixed** · [fixing incorrect docstring in upsampling.py](https://github.com/apache/tvm/issues/17845)
- ✅ **Fixed** *2 · [Fix docstring in batch_to_space_nd and bitpack](https://github.com/apache/tvm/issues/17848)



## 📈 Reproducing Coverage Results

1. Configure your local compiler build paths and `lcov_bin`.
2. Run the coverage script under the corresponding `experiments` directory.
3. Inspect the generated raw results in that experiment output directory.
4. Run the plotting script in the same compiler-specific experiment folder to regenerate the paper figures.

### Plotting coverage curves
> Both compiler directories include plotting scripts for turning the collected coverage data into figures:

- PyTorch plotting scripts: [`Torch/experiments/`](./Torch/experiments/)
- TVM plotting scripts: [`TVM/experiments/`](./TVM/experiments/)
