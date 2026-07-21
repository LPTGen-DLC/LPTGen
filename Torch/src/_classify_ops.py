import torch
from collections import defaultdict
import csv

# -------------------------------
# Step 1: Get all ATen operators
# -------------------------------
aten_ops = torch._C._jit_get_all_schemas()

# -------------------------------
# Step 2: Refined classification function
# -------------------------------
def classify_schema_corrected(schema):
    """More precise classification of ATen operators"""
    op_name = schema.name.lower()
    input_types = [str(arg.type) for arg in schema.arguments]
    output_types = [str(ret.type) for ret in schema.returns]

    # 1. Conv / Pool
    conv_pool_keywords = [
        'conv', 'pool', 'cudnn', 'mkldnn', 'mps', 'slow_conv', 'conv_depthwise', '_nnpack_spatial_convolution'
    ]
    if any(k in op_name for k in conv_pool_keywords):
        return 'conv_pool'

    # 2. View / Permute
    view_keywords = [
        'view', 'reshape', 'permute', 'transpose', 'flatten', 'squeeze', 'unsqueeze',
        'unflatten', 'stack', 'chunk', '_reshape', '_reshape_from_tensor', '_reshape_copy', '_reshape_alias_copy'
    ]
    if any(k in op_name for k in view_keywords):
        return 'view'

    # 3. Linear / Matmul
    linear_keywords = [
        'linear', 'mm', 'matmul', 'bmm', 'addmm', '_sparse_addmm', '_sparse_mm', 'sspaddmm',
        '_addmm_activation', 'fbgemm', 'quantized_linear', '_wrapped_linear'
    ]
    if any(k in op_name for k in linear_keywords):
        return 'linear'

    # 4. Reduction
    reduction_keywords = [
        'sum', 'mean', 'max', 'min', 'nanmean', 'nanmax', 'nanmin', 'amin', 'amax', 'prod', 'var', 'std', 'norm',
        'argmax', 'argmin', 'count_nonzero', 'cumprod', 'cumsum', 'cummax', 'cummin', 'histogram', 'histc',
        'segment_reduce', '_sparse_sum', '_sparse_csr_sum', '_sparse_csr_prod', '_sparse_log_softmax_backward_data',
        '_sparse_softmax_backward_data', 'native_norm', 'norm_except_dim', 'count'
    ]
    if any(x in op_name for x in reduction_keywords):
        return 'reduction'

    # Classify as reduction if output is scalar
    if any('int' in str(o.type) or 'float' in str(o.type) for o in schema.returns):
        return 'reduction'

    # 5. Pointwise: Tensor -> Tensor with broadcasting
    pointwise_exclude_keywords = (
        conv_pool_keywords + view_keywords + linear_keywords + [
            'dropout', 'feature', 'stft', 'glu', 'nan_to_num', 'fft', 'ctc_loss', 'pinverse',
            'quantized', 'prepacked', 'rngprims', 'prim', 'fake_quant', 'dequantize', 'quantize'
        ]
    )
    if input_types and output_types:
        if all('Tensor' in t for t in input_types) and all('Tensor' in t for t in output_types):
            if not any(k in op_name for k in pointwise_exclude_keywords):
                return 'pointwise'

    # 6. Fallback category
    return 'other'

# -------------------------------
# Step 3: Build operator metadata
# -------------------------------
op_info = {}
for schema in aten_ops:
    name = schema.name
    if not name.startswith('aten::'):
        continue

    op_info[name] = {
        'category': classify_schema_corrected(schema),
        'is_inplace': schema.name.endswith('_'),
        'inputs': [arg.name for arg in schema.arguments],
        'outputs': [ret.name for ret in schema.returns],
    }

# -------------------------------
# Step 4: Category statistics
# -------------------------------
category_count = defaultdict(int)
for info in op_info.values():
    category_count[info['category']] += 1

total_ops = len(op_info)

# -------------------------------
# Step 5: Export to CSV
# -------------------------------
csv_file = "aten_ops_classification.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['op_name', 'category', 'is_inplace', 'inputs', 'outputs'])
    for name, info in op_info.items():
        writer.writerow([
            name,
            info['category'],
            info['is_inplace'],
            ';'.join(info['inputs']),
            ';'.join(info['outputs'])
        ])