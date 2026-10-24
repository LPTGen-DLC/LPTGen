import random
from typing import List, Tuple

import numpy as np
import tvm
from tvm import te, tir, topi


def get_test_operators() -> List[Tuple[str, callable]]:
    """Return a list of (operator_name, constructor_function)."""

    def conv2d():
        a = te.placeholder((1, 3, 224, 224), name="A")
        w = te.placeholder((64, 3, 3, 3), name="W")
        return topi.nn.conv2d(a, w, stride=1, padding=1)

    def matmul():
        a = te.placeholder((1024, 1024), name="A")
        b = te.placeholder((1024, 1024), name="B")
        return topi.matmul(a, b)

    def relu():
        x = te.placeholder((1024,), name="X")
        return topi.nn.relu(x)

    return [("Conv2D", conv2d), ("MatMul", matmul), ("ReLU", relu)]


def get_tir_passes() -> List[Tuple[str, callable]]:
    """Return a list of (pass_name, pass_function)."""

    def apply_tile(sch: tir.Schedule):
        block = sch.get_block("compute")
        i, j = sch.get_loops(block)
        sch.tile(i, j, [32, 32])

    def apply_vectorize(sch: tir.Schedule):
        block = sch.get_block("compute")
        i, j = sch.get_loops(block)
        sch.vectorize(j)

    def apply_parallel(sch: tir.Schedule):
        block = sch.get_block("compute")
        i, j = sch.get_loops(block)
        sch.parallel(i)

    return [("Tile", apply_tile), ("Vectorize", apply_vectorize), ("Parallel", apply_parallel)]


def apply_random_tir_pass(sch: tir.Schedule) -> str:
    """Apply one random TIR pass and return its name."""
    pass_name, pass_func = random.choice(get_tir_passes())
    pass_func(sch)
    return pass_name


def run_diff_test(compute_func: callable, target: str = "llvm", dtype: str = "float32", rtol: float = 1e-3) -> bool:
    """Run differential testing and return whether the test passes."""
    input_shapes = [x.shape for x in compute_func.args]
    np_inputs = [np.random.rand(*shape).astype(dtype) for shape in input_shapes]
    tvm_inputs = [tvm.nd.array(x) for x in np_inputs]

    c = compute_func()
    schedule = te.create_schedule(c.op)
    mod = tvm.build(schedule, compute_func.args + [c], target=target)
    out = tvm.nd.array(np.zeros(c.shape, dtype=dtype))
    mod(*tvm_inputs, out)
    ref_output = out.numpy()

    sch = tir.Schedule(te.create_prim_func(compute_func()))
    pass_name = apply_random_tir_pass(sch)
    mod_opt = tvm.build(sch.mod["main"], compute_func.args + [c], target=target)
    out_opt = tvm.nd.array(np.zeros(c.shape, dtype=dtype))
    mod_opt(*tvm_inputs, out_opt)
    opt_output = out_opt.numpy()

    try:
        np.testing.assert_allclose(ref_output, opt_output, rtol=rtol)
        return True
    except AssertionError:
        print(f"Differential test failed. Pass={pass_name}, MaxError={np.max(np.abs(ref_output - opt_output))}")
        return False


def test_all_operators():
    """Run the demo differential tests for all operators."""
    operators = get_test_operators()
    targets = ["llvm", "cuda"]
    passed = 0
    total = 0

    for op_name, op_func in operators:
        for target in targets:
            total += 1
            print(f"Testing operator: {op_name}, Target: {target}")
            success = run_diff_test(op_func, target=target)
            if success:
                passed += 1
                print("Passed")
            else:
                print("Failed")

    print(f"\nCompleted: {passed}/{total} passed")


if __name__ == "__main__":
    test_all_operators()
