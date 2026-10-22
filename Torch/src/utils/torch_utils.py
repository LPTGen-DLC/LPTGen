import torch
import ast

from scipy.sparse.tests.test_coo import dot_shapes

dtypes = [torch.float16, torch.bfloat16]
device = torch.device("cuda")
dtype_map = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    'bool': torch.bool,
}


def is_true_str(s: str) -> bool:
    if s.startswith("torch"):
        return False
    try:
        ast.literal_eval(s)
        return False
    except (ValueError, SyntaxError):
        return True


def generate_api_inputs(para_config):
    def _is_tensor(value):
        if isinstance(value, (list, tuple)) :
            if len(value)>=2 and isinstance(value[1], str) and value[1] in dtype_map.keys():  # [[2, 3, 4], "bfloat16"]
                return True
            elif len(value)>=2 and isinstance(value[1], str) and value[1].startswith('torch.'):  # [[2, 3, 4], "torch.bfloat16"]
                return True
            elif len(value) >=2 and value[1] in dtype_map.values():  # [[2, 3, 4], torch.bfloat16]
                return True
            elif len(value)>=1 and isinstance(value[0], (list, tuple)):
                return _is_tensor(value[0])
        elif isinstance(value, dict):
            if len(value.keys()) == 2 and list(value.keys()) == ['shape', 'dtype']:
                return True
        return False

    def _make_high_dim_tensor(value):
        def _make_tensor(shape, dtype_str):
            if not isinstance(dtype_str, str):
                dtype_str = str(dtype_str)
            if isinstance(dtype_str, str) and 'torch.' not in dtype_str and dtype_str in dtype_map.keys():
                dtype_str = str(dtype_map[dtype_str])
            if "uint8" in dtype_str:
                return f"torch.randint(0, 128, {shape},dtype={dtype_str}, device='{device}')"
            elif 'int' in dtype_str:
                return f"torch.randint(0, 3, {shape},dtype={dtype_str}, device='{device}')"  # use (0,2) to keep valid of 'index' params
            elif 'bool' in dtype_str:
                return f"torch.randint(0, 2, {shape}, dtype={dtype_str}, device='{device}')"
            else:
                return f"torch.randn({shape},dtype={dtype_str}, device='{device}')"

        if isinstance(value, (list, tuple)):
            if len(value) == 2 and isinstance(value[1], str) and value[1] in dtype_map.keys():  # [[2, 3, 4], "bfloat16"]
                return _make_tensor(v[0], v[1])
            elif len(value) == 2 and isinstance(value[1], str) and value[1].startswith('torch.'): # [[2,], "torch.float16"]
                return _make_tensor(v[0], v[1])
            elif len(value) == 2 and value[1] in dtype_map.values():
                return _make_tensor(v[0], str(v[1]))
            elif len(value) >= 1 and isinstance(value[0], (list, tuple)):
                high_dim_tensor = []
                for item in value:
                    if _is_tensor(item):
                        if len(item) == 2:
                            high_dim_tensor.append(_make_tensor(item[0], item[1]))
                        elif len(item) ==1 and _is_tensor(item[0]):
                            high_dim_tensor.append(_make_tensor(item[0][0], item[0][1]))
                return high_dim_tensor
        elif isinstance(value, dict):
            return _make_tensor(v['shape'], v['dtype'])
        return None

    new_config = {}
    for k, v in para_config.items():
        if _is_tensor(v):
            new_config[k] = _make_high_dim_tensor(v)
        else:
            new_config[k] = v
    return new_config



def get_callable_from_aten(op_name: str):
    try:
        obj = getattr(torch.ops.aten, op_name)
    except AttributeError:
        return None

    if hasattr(obj, "default") and callable(obj.default):  # Prefer default overload
        return obj.default

    if callable(obj):  # Fallback to any callable overload
        return obj
    return None


if __name__ == '__main__':
    input =  [{"self": [([], "float64"), ([2, 2, 2, 2], "float32")]}][-1]
    input = {"indices": [[1, 8], "torch.int64"], "values": [[8], torch.int64], "size": [10, 10], "dtype": torch.int64, "layout": torch.sparse_coo, "device": "cpu", "pin_memory": None, "is_coalesced": False}
    print(input)
    res = generate_api_inputs(input)
    print(res)
