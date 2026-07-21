import inspect
import random
from typing import List, Tuple, Callable
import tvm
from tvm import topi, tir


def get_topi_operators(namespace) -> List[Tuple[str, Callable]]:
    """Automatically discover all callable operators in a topi namespace"""
    operators = []
    for name, func in inspect.getmembers(eval(namespace), inspect.isfunction):
        if name.startswith("_"):  # Skip private functions
            continue
        if name.startswith("get_const_tuple"):
            continue
        try:
            operators.append((namespace+"."+name, func))   #_ = func(*[te.placeholder((32, 32)) for _ in sig.parameters])
        except Exception as e:
            print(f"[DEBUG]: {e}")
            continue
    return operators


def get_all_tir_schedules() -> List[Tuple[str, Callable]]:
    """Discover all available TIR passes and schedule primitives"""
    schedules = []
    # Schedule primitives
    for name, func in inspect.getmembers(tir.schedule.Schedule, inspect.isfunction):
        if not name.startswith("_") and name != "get_current_render":
            print("schedule." + name)
            schedules.append(("schedule." + name, func))
    return schedules


def select_transforms():
    pass_ctx = tvm.transform.PassContext.current()
    config = pass_ctx.config
    passes = [
        "tir.transform.LowerCrossThreadReduction()",
        "tir.transform.LowerInitBlock()",
        "tir.transform.PlanAndUpdateBufferAllocationLocation()",
        "tir.transform.ConvertBlocksToOpaque()",
        "tir.transform.LiftThreadBinding()",
        "tir.transform.ManifestSharedMemoryLocalStage()",
        "tir.transform.CompactBufferAllocation()",
        "tir.transform.LowerAutoCopy()",
        "tir.transform.UnifyThreadBinding()",
        "tir.transform.LowerMatchBuffer()",
        "tir.transform.Simplify()",
        "tir.transform.InjectPermutedLayout()",
        "tir.transform.InjectSoftwarePipeline()",
        "tir.transform.TransformMmaBufferLayout()",
        "tir.transform.LowerOpaqueBlock()",
        "tir.transform.FlattenBuffer()",
        # "tir.transform.BF16ComputeLegalize()",
        "tir.transform.NarrowDataType(32)",
        "tir.transform.LoopPartition()",
        "tir.transform.VectorizeLoop(True)",
        "tir.transform.InjectVirtualThread()",
        "tir.transform.InjectDoubleBuffer()",
    ]

    if not bool(config.get("tir.disable_storage_rewrite", False)):
        passes.append("tir.transform.StorageRewrite()")
    if config.get("tir.use_async_copy", False):
        passes.append("tir.transform.LowerAsyncDMA()")
    passes.extend(
        [
            "tir.transform.HoistIfThenElse()",
            "tir.transform.UnrollLoop()",
            "tir.transform.RenormalizeSplitPattern()",
            "tir.transform.Simplify()",
            "tir.transform.RemoveNoOp()",
            "tir.transform.RewriteUnsafeSelect()",
        ]
    )

    # Additional passes based on configuration.
    if bool(config.get("tir.instrument_bound_checkers", False)):
        passes.append("tir.transform.InstrumentBoundCheckers()")
    if bool(config.get("tir.ptx_ldg32", False)):
        passes.append("tir.transform.InjectPTXLDG32(True)")
    passes.append("tir.transform.CommonSubexprElimTIR(True, True)")
    if bool(config.get("tir.instrument_lwp", False)):
        passes.append("tir.transform.InstrumentProfileIntrinsics()")

    # passes = pick_n_passes(passes, 10)
    passes.extend(
        [
            # Bind the target first so that target-specific attributes are available.
            # "tir.transform.FP8ComputeLegalize()",
            # VerifyVTCMLimit must occur before LowerVtcmAlloc.
            "tir.transform.VerifyVTCMLimit()",
            "tir.transform.LowerVtcmAlloc()",
            "tir.transform.VerifyMemory()",
            "tir.transform.AnnotateEntryFunc()",
        ]
    )
    if bool(config.get("tir.detect_global_barrier", False)):
        passes.append(tir.transform.ThreadSync("global"))
    passes.extend(
        [
            "tir.transform.ThreadSync('shared')",
            "tir.transform.ThreadSync('shared.dyn')",
            "tir.transform.ThreadSync('warp')",
            "tir.transform.InferFragment()",
            "tir.transform.LowerThreadAllreduce()",
        ]
    )
    if bool(config.get("tir.use_async_copy", False)):
        passes.append("tir.transform.InjectPTXAsyncCopy()")
    if bool(config.get("tir.ptx_ldg32", False)):
        passes.append("tir.transform.InjectPTXLDG32()")
    passes.extend(
        [
            "tir.transform.AnnotateDeviceRegions()",
            # MergeSharedMemoryAllocations must follow SplitHostDevice.
            "tir.transform.SplitHostDevice()",
            "tir.transform.MergeSharedMemoryAllocations()",
            "tir.transform.MakePackedAPI()",
            # "tir.transform.FP8StorageLegalize()",
            # "tir.transform.BF16StorageLegalize()",
            "tir.transform.LowerDeviceKernelLaunch()",
        ]
    )
    return passes


def pick_n_passes(pass_list, n):
    selected_passes = random.choices(pass_list, k=n)
    final_passes = []
    for i, _pass in enumerate(selected_passes):
        final_passes.append(_pass)

        if _pass == tir.transform.LowerVtcmAlloc() and selected_passes[i+1] != tir.transform.VerifyVTCMLimit():
            final_passes.append(tir.transform.VerifyVTCMLimit())
        elif _pass == tir.transform.SplitHostDevice() and selected_passes[i+1] != tir.transform.MergeSharedMemoryAllocations():
            final_passes.append(tir.transform.MergeSharedMemoryAllocations())
    return final_passes


if __name__ == '__main__':
    import tvm
    from tvm import te, topi, tir

    data = te.placeholder((1, 3, 224, 224), dtype='float32', name='data')
    op_config = {'data': data, 'output_size': (112, 112), 'pool_type': 'max', 'layout': 'NCHW', }
    op_output = topi.nn.adaptive_pool(**op_config)
    sch = tir.Schedule(te.create_prim_func([data, op_output]).with_attr("target", tvm.target.Target("llvm")))
    # mod = tir.transform.MakePackedAPI()(sch.mod)
    # print(mod)
    # mod = tir.transform.FP8StorageLegalize()(mod)
    #
    select_transforms()
