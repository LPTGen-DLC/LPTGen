import torch


def categorize_aten_operator(op_name, schema=None, input_types=None, output_types=None):
    op_name = op_name.lower()

    conv_pool_keywords = ["conv", "pool", "cudnn", "mkldnn", "mps", "slow_conv", "_nnpack_spatial_convolution"]
    if any(keyword in op_name for keyword in conv_pool_keywords):
        return "conv_pool"

    view_keywords = [
        "view", "reshape", "permute", "transpose", "flatten", "squeeze", "unsqueeze",
        "unflatten", "stack", "chunk", "_reshape", "_reshape_from_tensor", "_reshape_copy",
        "_reshape_alias_copy",
    ]
    if any(keyword in op_name for keyword in view_keywords):
        return "view"

    linear_keywords = [
        "linear", "mm", "matmul", "bmm", "addmm", "_sparse_addmm", "_sparse_mm",
        "sspaddmm", "_addmm_activation", "fbgemm", "quantized_linear", "_wrapped_linear",
    ]
    if any(keyword in op_name for keyword in linear_keywords):
        return "linear"

    reduction_keywords = [
        "sum", "mean", "max", "min", "nanmean", "nanmax", "nanmin", "amin", "amax", "prod", "var",
        "std", "norm", "argmax", "argmin", "count_nonzero", "cumprod", "cumsum", "cummax", "cummin",
        "histogram", "histc", "segment_reduce", "_sparse_sum", "_sparse_csr_sum", "_sparse_csr_prod",
        "_sparse_log_softmax_backward_data", "_sparse_softmax_backward_data", "native_norm",
        "norm_except_dim", "count",
    ]
    if any(keyword in op_name for keyword in reduction_keywords):
        return "reduction"

    pointwise_exclude_keywords = conv_pool_keywords + view_keywords + linear_keywords + [
        "dropout", "feature", "stft", "glu", "nan_to_num", "fft", "ctc_loss", "pinverse", "quantized",
        "prepacked", "rngprims", "prim", "fake_quant", "dequantize", "quantize",
    ]
    if input_types and output_types:
        if all("Tensor" in t for t in input_types) and all("Tensor" in t for t in output_types):
            if not any(keyword in op_name for keyword in pointwise_exclude_keywords):
                return "pointwise"

    return "other"


def build_op_info():
    schemas = torch._C._jit_get_all_schemas()
    op_info = {}
    for schema in schemas:
        name = schema.name
        if not name.startswith("aten"):
            continue
        if name.startswith("aten::_"):
            continue

        input_types = [str(arg.type) for arg in schema.arguments]
        output_types = [str(arg.type) for arg in schema.returns]
        if not output_types or "Tensor" not in output_types[0]:
            continue

        category = categorize_aten_operator(name, schema, input_types, output_types)
        fusion_compatible = category == "pointwise" and not name.endswith("_")
        op_info[name] = {
            "category": category,
            "fusion_compatible": fusion_compatible,
            "inputs": input_types,
            "outputs": output_types,
            "signature": schema,
        }
    return op_info


def get_fusion_candidates(op_name, op_info):
    if op_name not in op_info:
        return []

    info = op_info[op_name]
    if not info["fusion_compatible"]:
        return []

    candidates = []
    for other_name, other_info in op_info.items():
        if other_name == op_name:
            continue
        if not other_info["fusion_compatible"]:
            continue
        if all("Tensor" in t for t in info["outputs"]) and all("Tensor" in t for t in other_info["inputs"]):
            candidates.append((other_name, other_info["signature"]))

    return candidates


if __name__ == "__main__":
    op_info = build_op_info()
    example_op = "aten::special_bessel_y1"
    fusion_list = get_fusion_candidates(example_op, op_info)
    print(f"Fusion candidate count for '{example_op}': {len(fusion_list)}")
    for item in fusion_list[:200]:
        print(item[0], item[1])
