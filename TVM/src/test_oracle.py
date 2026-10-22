"""Optional TOPI Python references; unverified mappings use the legacy oracle.

Verified against apache/tvm commit 36f2502f, python/tvm/topi/{nn,testing}.
TODO: Extend this explicit registry only after checking operator/reference semantics.
Generated tests import this helper from their own directory.
"""
import copy
import importlib
import inspect
import os

import numpy as np


class ReferenceUnavailable(ValueError):
    pass


class NondeterministicReference(ValueError):
    pass


# No suffix/name guessing: these are reviewed operator-to-reference relationships.
# TODO: Verify conv/pool/normalization adapters and mixed-precision/layout contracts
# before expanding coverage. Their mere presence in testing is not a mapping.
_REFERENCE_NAMES = {
    "topi.nn.softmax": "softmax_python",
    "topi.nn.log_softmax": "log_softmax_python",
    "topi.nn.dense": "dense",
    "topi.nn.batch_matmul": "batch_matmul",
}
_REFERENCE_PARAMETERS = {
    "softmax_python": {"a_np", "axis"},
    "log_softmax_python": {"a_np", "axis"},
    "dense": {"x", "y", "bias", "use_bias", "use_relu", "out_dtype"},
    "batch_matmul": {"x", "y", "out_dtype", "trans_x", "trans_y"},
}


def _floating_tensor(value, rank=None):
    if (not isinstance(value, np.ndarray) or value.dtype.name not in {"float32", "float64"}
            or (rank is not None and value.ndim != rank) or not np.isfinite(value).all()):
        raise ReferenceUnavailable("tensor dtype, rank or values outside verified reference range")
    return value


def _plain_layout(arguments):
    for name in ("auto_scheduler_rewritten_layout", "meta_schedule_original_shape"):
        value = arguments.get(name)
        if value is not None and value != "":
            raise ReferenceUnavailable("rewritten layouts are not covered by this reference")


def _same_output_dtype(arguments, data):
    dtype = arguments.get("out_dtype")
    if dtype is not None and dtype != data.dtype.name:
        raise ReferenceUnavailable("mixed precision is not covered by this reference")


def _reference_arguments(op_name, arguments):
    if op_name in {"topi.nn.softmax", "topi.nn.log_softmax"}:
        if set(arguments) - {"x", "axis"}:
            raise ReferenceUnavailable("operator signature changed")
        data = _floating_tensor(arguments["x"])
        axis = arguments["axis"]
        if (not isinstance(axis, (int, np.integer)) or isinstance(axis, (bool, np.bool_))
                or not -data.ndim <= axis < data.ndim or data.shape[axis] == 0):
            raise ReferenceUnavailable("axis outside verified reference range")
        # TOPI defaults to -1, whereas testing.softmax_python defaults to 1.
        return (data,), {"axis": int(axis)}

    _plain_layout(arguments)
    if op_name == "topi.nn.dense":
        if set(arguments) - {"data", "weight", "bias", "out_dtype",
                             "auto_scheduler_rewritten_layout", "meta_schedule_original_shape"}:
            raise ReferenceUnavailable("operator signature changed")
        data = _floating_tensor(arguments["data"], 2)
        weight = _floating_tensor(arguments["weight"], 2)
        if data.dtype != weight.dtype or data.shape[1] != weight.shape[1]:
            raise ReferenceUnavailable("dense input contract not covered")
        _same_output_dtype(arguments, data)
        bias = arguments.get("bias")
        if bias is not None:
            _floating_tensor(bias, 1)
            if bias.dtype != data.dtype or bias.shape != (weight.shape[0],):
                raise ReferenceUnavailable("dense bias contract not covered")
        # testing.dense does not enable bias merely because a bias was passed.
        return (data, weight, bias), {"use_bias": bias is not None, "use_relu": False,
                                      "out_dtype": arguments.get("out_dtype")}

    if set(arguments) - {"tensor_a", "tensor_b", "oshape", "out_dtype", "transpose_a",
                         "transpose_b", "auto_scheduler_rewritten_layout", "meta_schedule_original_shape"}:
        raise ReferenceUnavailable("operator signature changed")
    a = _floating_tensor(arguments["tensor_a"], 3)
    b = _floating_tensor(arguments["tensor_b"], 3)
    ta, tb = arguments["transpose_a"], arguments["transpose_b"]
    if not isinstance(ta, (bool, np.bool_)) or not isinstance(tb, (bool, np.bool_)):
        raise ReferenceUnavailable("transpose flags are not boolean")
    if (arguments.get("oshape") is not None or a.dtype != b.dtype
            or a.shape[1 if ta else 2] != b.shape[2 if tb else 1]
            or not (a.shape[0] == b.shape[0] or a.shape[0] == 1 or b.shape[0] == 1)):
        raise ReferenceUnavailable("batch_matmul shape/dtype contract not covered")
    _same_output_dtype(arguments, a)
    return (a, b), {"out_dtype": arguments.get("out_dtype"), "trans_x": bool(ta), "trans_y": bool(tb)}


def _reference_outputs(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    if len(values) != 1 or not isinstance(values[0], np.ndarray):
        raise ReferenceUnavailable("unexpected reference output structure")
    # Snapshot outputs, since a second call could otherwise modify an earlier view.
    return [values[0].copy()]


def assert_reference_outputs(expected, actual):
    """Never broadcast or silently truncate outputs; comparison errors must escape."""
    if len(expected) != len(actual):
        raise AssertionError("An inconsistency detected: output count mismatch")
    for index, (left, right) in enumerate(zip(expected, actual)):
        right = right.numpy() if hasattr(right, "numpy") else np.asarray(right)
        if left.shape != right.shape:
            raise AssertionError(f"An inconsistency detected: output {index} shape {left.shape} != {right.shape}")
        if left.dtype.kind in "biu" or right.dtype.kind in "biu":
            np.testing.assert_array_equal(left, right, err_msg="An inconsistency detected.")
        else:
            np.testing.assert_allclose(left, right, rtol=1e-3, atol=1e-3,
                                       err_msg="An inconsistency detected.")


def try_reference(op_name, input_config, numpy_inputs):
    """Return (outputs, reason); None means keep the existing compiled-path oracle.

    Only preparation/reference execution errors cause fallback. Comparisons with
    compiler outputs happen outside this function and must never cause fallback.
    """
    if os.getenv("LPTGEN_TVM_REFERENCE", "1").lower() in {"0", "false", "off"}:
        return None, "independent reference disabled"
    if op_name not in _REFERENCE_NAMES:
        return None, "no verified operator-to-reference mapping"
    try:
        import tvm.topi.testing as testing
        reference = getattr(testing, _REFERENCE_NAMES[op_name], None)
        if not callable(reference):
            raise ReferenceUnavailable("corresponding tvm.topi.testing reference is absent")
        topi = importlib.import_module("tvm.topi")
        operator = topi
        for part in op_name.split(".")[1:]:
            operator = getattr(operator, part)
        # Generated model_func passes config values positionally in this order.
        values = [numpy_inputs[key] if key in numpy_inputs else value
                  for key, value in input_config.items()]
        call = inspect.signature(operator).bind(*values)
        call.apply_defaults()
        bound = call.arguments
        args, kwargs = _reference_arguments(op_name, bound)
        # Validate the installed version before calling: existence alone is insufficient.
        signature = inspect.signature(reference)
        if set(signature.parameters) != _REFERENCE_PARAMETERS[_REFERENCE_NAMES[op_name]]:
            raise ReferenceUnavailable("reference signature differs from the verified adapter")
        signature.bind(*args, **kwargs)
        first = _reference_outputs(reference(*copy.deepcopy(args), **copy.deepcopy(kwargs)))
        second = _reference_outputs(reference(*copy.deepcopy(args), **copy.deepcopy(kwargs)))
        try:
            assert_reference_outputs(first, second)
        except AssertionError as exc:
            raise NondeterministicReference("independent reference outputs differ") from exc
        return first, "tvm.topi.testing." + _REFERENCE_NAMES[op_name]
    except NondeterministicReference:
        raise
    except Exception as exc:
        return None, f"reference unavailable ({type(exc).__name__}): {exc}"
