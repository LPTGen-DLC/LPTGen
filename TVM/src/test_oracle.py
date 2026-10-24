"""Optional TOPI testing references; unsupported calls use the legacy oracle."""
import copy
import importlib
import inspect
import os

import numpy as np


class ReferenceUnavailable(ValueError):
    pass


class NondeterministicReference(ValueError):
    pass


# Correspondences follow TVM's tests/python/topi/python tests (v0.14.0).
_REFERENCE_NAMES = {
    "topi.nn.softmax": "softmax_python",
    "topi.nn.log_softmax": "log_softmax_python",
    "topi.nn.dense": "dense",
    "topi.nn.batch_matmul": "batch_matmul",
    "topi.nn.batch_norm": "batch_norm",
    "topi.nn.layer_norm": "layer_norm_python",
    "topi.nn.group_norm": "group_norm_python",
    "topi.nn.instance_norm": "instance_norm_python",
    "topi.nn.rms_norm": "rms_norm_python",
    "topi.nn.l2_normalize": "l2_normalize_python",
    "topi.nn.lrn": "lrn_python",
    "topi.nn.dilate": "dilate_python",
    "topi.nn.space_to_batch_nd": "space_to_batch_nd_python",
    "topi.nn.batch_to_space_nd": "batch_to_space_nd_python",
    "topi.nn.depth_to_space": "depth_to_space_python",
    "topi.nn.space_to_depth": "space_to_depth_python",
    "topi.nn.nll_loss": "nll_loss",
    "topi.gather": "gather_python",
    "topi.gather_nd": "gather_nd_python",
    "topi.one_hot": "one_hot",
    "topi.matrix_set_diag": "matrix_set_diag",
    "topi.sequence_mask": "sequence_mask",
    "topi.strided_slice": "strided_slice_python",
    "topi.strided_set": "strided_set_python",
    "topi.searchsorted": "searchsorted_ref",
    "topi.image.affine_grid": "affine_grid_python",
    "topi.image.grid_sample": "grid_sample_python",
    "topi.image.crop_and_resize": "crop_and_resize_python",
    "topi.vision.reorg": "reorg_python",
    "topi.vision.roi_align_nchw": "roi_align_nchw_python",
    "topi.vision.roi_align_nhwc": "roi_align_nhwc_python",
    "topi.vision.roi_pool_nchw": "roi_pool_nchw_python",
    "topi.nn.correlation_nchw": "correlation_nchw_python",
    "topi.nn.conv1d_ncw": "conv1d_ncw_python",
    "topi.nn.group_conv1d_ncw": "group_conv1d_ncw_python",
    "topi.nn.conv2d_nchw": "conv2d_nchw_python",
    "topi.nn.conv2d_nhwc": "conv2d_nhwc_python",
    "topi.nn.conv2d_hwcn": "conv2d_hwcn_python",
    "topi.nn.group_conv2d_nchw": "conv2d_nchw_python",
    "topi.nn.group_conv2d_nhwc": "conv2d_nhwc_python",
    "topi.nn.conv3d_ncdhw": "conv3d_ncdhw_python",
    "topi.nn.conv3d_ndhwc": "conv3d_ndhwc_python",
    "topi.nn.depthwise_conv2d_nchw": "depthwise_conv2d_python_nchw",
    "topi.nn.depthwise_conv2d_nhwc": "depthwise_conv2d_python_nhwc",
    "topi.nn.conv1d_transpose_ncw": "conv1d_transpose_ncw_python",
    "topi.nn.conv2d_transpose_nchw": "conv2d_transpose_nchw_python",
    "topi.nn.group_conv2d_transpose_nchw": "conv2d_transpose_nchw_python",
    "topi.nn.conv3d_transpose_ncdhw": "conv3d_transpose_ncdhw_python",
    "topi.nn.deformable_conv2d_nchw": "deformable_conv2d_nchw_python",
    "topi.nn.deformable_conv2d_nhwc": "deformable_conv2d_nhwc_python",
    **{f"topi.nn.pool{n}d": "poolnd_python" for n in (1, 2, 3)},
    "topi.nn.adaptive_pool": "adaptive_pool",
    "topi.nn.adaptive_pool3d": "adaptive_pool",
    "topi.nn.pool_grad": "pool_grad_nchw",
    **{f"topi.image.resize{n}d": f"resize{n}d_python" for n in (1, 2, 3)},
}
# TODO: Verify attention, LSTM, backward-weight, packed layouts and non-trailing norm axes.


def _reference_arguments(op_name, arguments, testing):
    a = dict(arguments)
    for key in ("auto_scheduler_rewritten_layout", "meta_schedule_original_shape",
                "meta_schedule_origin_shape"):
        if a.pop(key, None) not in (None, ""):
            raise ReferenceUnavailable("rewritten layouts are unsupported")
    name = op_name.rsplit(".", 1)[-1]
    if name in {"layer_norm", "instance_norm", "rms_norm"}:
        rank = a["data"].ndim
        axes = a["axis"] if isinstance(a["axis"], (tuple, list)) else (a["axis"],)
        axes = tuple(int(axis) % rank for axis in axes)
        if axes != tuple(range(rank - len(axes), rank)):
            raise ReferenceUnavailable("normalization reference requires trailing axes")
    elif name == "group_norm":
        rank = a["data"].ndim
        a["channel_axis"] %= rank
        a["axes"] = tuple(int(axis) % rank for axis in a["axes"])
    if name == "dense":
        data, weight, bias, dtype = a.values()
        dtype = dtype or data.dtype
        bias = None if bias is None else bias.astype(dtype)
        return (data, weight, bias), {"use_bias": bias is not None, "out_dtype": dtype}
    if name == "batch_matmul":
        data, weight, shape, dtype, ta, tb = a.values()
        if shape is not None:
            raise ReferenceUnavailable("explicit batch_matmul output shape is unsupported")
        return (data, weight), {"out_dtype": dtype, "trans_x": ta, "trans_y": tb}
    if name == "batch_norm":
        defaults = dict(axis=1, epsilon=1e-5, center=True, scale=True, training=False, momentum=0.1)
        for key, value in defaults.items():
            if a[key] is None:
                a[key] = value
    elif name == "lrn":
        if a["axis"] not in (1, 3) or a["data"].ndim != 4:
            raise ReferenceUnavailable("LRN reference supports 4D axes 1 and 3")
        return (a["data"], a["size"], a["axis"], a["bias"], a["alpha"], a["beta"]), {}
    elif name == "strided_slice":
        return (a["a"], a["begin"], a["end"], a["strides"]), {
            "axes": a["axes"], "slice_mode": a["slice_mode"]}
    elif name == "dilate":
        a.pop("name", None)
    elif name in {"depth_to_space", "space_to_depth"}:
        if a.pop("layout") != "NCHW":
            raise ReferenceUnavailable("space/depth reference requires NCHW")
        # Older references use height for both spatial dimensions.
        if a["data"].shape[2] != a["data"].shape[3]:
            raise ReferenceUnavailable("non-square space/depth reference is unverified")
    elif name.startswith("roi_align_"):
        return (a["data"], a["rois"], a["pooled_size"], a["spatial_scale"],
                a["sample_ratio"], a["mode"]), {}
    elif name == "crop_and_resize":
        dtype = a.pop("out_dtype")
        if dtype not in (None, a["data"].dtype.name):
            raise ReferenceUnavailable("crop_and_resize output cast is unsupported")
        if a["extrapolation_value"] is None or min(a["crop_size"]) <= 1:
            raise ReferenceUnavailable("crop extrapolation or singleton size is unverified")
    elif name.startswith("resize"):
        data, roi, size = a["data"], a["roi"], a["size"]
        n = int(name[6])
        layout = a["layout"]
        spatial = {1: {"NCW": (2,), "NWC": (1,)},
                   2: {"NCHW": (2, 3), "NHWC": (1, 2)},
                   3: {"NCDHW": (2, 3, 4), "NDHWC": (1, 2, 3)}}[n][layout]
        if (np.any(roi) or a["rounding_method"] != "" or a["bicubic_alpha"] != -0.5
                or a["bicubic_exclude"] != 0 or a["extrapolation_value"] != 0
                or a["output_shape"] is not None
                or a["out_dtype"] not in (None, data.dtype.name)):
            raise ReferenceUnavailable("non-default resize options are unsupported")
        scale = tuple(int(size[i]) / data.shape[axis] for i, axis in enumerate(spatial))
        return (data, scale, layout, a["method"], a["coordinate_transformation_mode"]), {}
    elif name in {"pool1d", "pool2d", "pool3d"}:
        n = int(name[4])
        padding = a["padding"]
        if isinstance(padding, (int, np.integer)):
            padding = [padding] * (2 * n)
        if len(padding) == n:
            padding = list(padding) * 2
        if len(padding) != 2 * n:
            raise ReferenceUnavailable("unsupported pool padding")
        return (a["data"], a["kernel"], a["stride"], a["dilation"],
                padding[:n], padding[n:], a["pool_type"]), {
                    "count_include_pad": a["count_include_pad"], "ceil_mode": a["ceil_mode"],
                    "dtype": a["data"].dtype.name, "layout": a["layout"]}
    elif name == "pool_grad":
        if a.pop("layout") != "NCHW":
            raise ReferenceUnavailable("pool_grad reference requires NCHW")
        return (a["data"], a["grads"], a["kernel"], a["stride"], a["padding"],
                a["pool_type"], a["ceil_mode"], a["count_include_pad"]), {}
    elif "conv" in name:
        dtype = a.pop("out_dtype", None) or next(iter(a.values())).dtype.name
        if dtype not in {"float32", "float64"}:
            raise ReferenceUnavailable("convolution precision is unverified")
        # TOPI casts operands before multiplication; references need the same cast.
        for key, value in a.items():
            if isinstance(value, np.ndarray) and key != "offset":
                a[key] = value.astype(dtype)
        if name == "conv1d_transpose_ncw" and np.isscalar(a["output_padding"]):
            a["output_padding"] = (a["output_padding"],)
        if "transpose" in name or "deformable" in name:
            return tuple(a.values()), {}
        data, weight = list(a.values())[:2]
        dilation = a.pop("dilation")
        if "conv1d" in name:
            return tuple(a.values())[:4] + (dilation,) + tuple(a.values())[4:], {}
        layout = name.rsplit("_", 1)[-1]
        axes = {"nchw": (2, 3), "nhwc": (0, 1), "hwcn": (0, 1),
                "ncdhw": (2, 3, 4), "ndhwc": (0, 1, 2)}[layout]
        if "depthwise" in name and layout == "nhwc":
            kernel_layout = a.pop("kernel_layout")
            if kernel_layout == "HWOI":
                weight = weight.transpose(0, 1, 3, 2)
            elif kernel_layout != "HWIO":
                raise ReferenceUnavailable("unsupported depthwise kernel layout")
        factors = [1] * weight.ndim
        dilation = [dilation] * len(axes) if np.isscalar(dilation) else dilation
        for axis, factor in zip(axes, dilation):
            factors[axis] = int(factor)
        weight = testing.dilate_python(weight, factors)
        return (data, weight) + tuple(a.values())[2:], {}
    return tuple(a.values()), {}


def _reference_outputs(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(not isinstance(x, (np.ndarray, np.generic)) for x in values):
        raise ReferenceUnavailable("unexpected reference output structure")
    return [np.asarray(x).copy() for x in values]


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
    """Reference preparation failures fall back; compiler mismatches must escape."""
    if os.getenv("LPTGEN_TVM_REFERENCE", "1").lower() in {"0", "false", "off"}:
        return None, "independent reference disabled"
    canonical_name = (op_name or "").replace("topi.transform.", "topi.")
    generic_conv = canonical_name in {"topi.nn.conv1d", "topi.nn.conv2d"}
    if canonical_name not in _REFERENCE_NAMES and not generic_conv:
        return None, "no operator-to-reference mapping"
    try:
        import tvm.topi.testing as testing
        operator = importlib.import_module("tvm.topi")
        for part in op_name.split(".")[1:]:
            operator = getattr(operator, part)
        # Match the generated model's positional call, including TOPI defaults.
        values = [numpy_inputs[key] if key in numpy_inputs else value
                  for key, value in input_config.items()]
        call = inspect.signature(operator).bind(*values)
        call.apply_defaults()
        if generic_conv:
            layout = call.arguments.pop("data_layout")
            kernel_layout = call.arguments.pop("kernel_layout")
            expected_layout = {"NCW": "OIW", "NCHW": "OIHW", "NHWC": "HWIO", "HWCN": "HWIO"}[layout]
            if kernel_layout not in ("", expected_layout):
                raise ReferenceUnavailable("non-default convolution kernel layout is unsupported")
            canonical_name += "_" + layout.lower()
        reference_name = _REFERENCE_NAMES.get(canonical_name)
        reference = getattr(testing, reference_name, None) if reference_name else None
        if not callable(reference):
            raise ReferenceUnavailable("corresponding tvm.topi.testing reference is absent")
        for value in numpy_inputs.values():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ReferenceUnavailable("non-finite reference inputs are unverified")
        if canonical_name == "topi.gather_nd":
            data, indices = call.arguments.values()
            if ((data.dtype.kind in "iu" and np.any(np.abs(data.astype(object)) > 2**53))
                    or np.any(indices < -(2**31)) or np.any(indices >= 2**31)):
                raise ReferenceUnavailable("gather_nd reference loses integer precision")
        args, kwargs = _reference_arguments(canonical_name, call.arguments, testing)
        inspect.signature(reference).bind(*args, **kwargs)
        first = _reference_outputs(reference(*copy.deepcopy(args), **copy.deepcopy(kwargs)))
        second = _reference_outputs(reference(*copy.deepcopy(args), **copy.deepcopy(kwargs)))
        try:
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)
            if len(first) != len(second):
                raise AssertionError("reference output count changed")
        except AssertionError as exc:
            raise NondeterministicReference("independent reference outputs differ") from exc
        return first, "tvm.topi.testing." + reference_name
    except NondeterministicReference:
        raise
    except Exception as exc:
        return None, f"reference unavailable ({type(exc).__name__}): {exc}"
