import os.path
import random
import torch
from utils import torch_utils
from utils.llm_utils import call_llm
import extract_lowering_conditions
import my_obj
import mutate_params

import get_fusion_rules
import mutate4opfusion

random.seed(2026)


seed_str_header = '''import sys
import os
import torch
# os.environ['TORCHINDUCTOR_EMULATE_PRECISION_CASTS']= '1'
'''

seed_str_foot = '''
try:
    compiled_eager = torch.compile(model_func, backend="eager")
    out1 = compiled_eager(**input_config)
    out2 = compiled_eager(**input_config)
    torch.testing.assert_close(out1, out2)
except Exception as e:
    print(e)
    sys.exit(0)
compiled_inductor = torch.compile(model_func, backend="inductor", )  # options={"trace.enabled": True,}
out_inductor = compiled_inductor(**input_config)
torch.testing.assert_close(out1, out_inductor)
'''

def get_all_operators():
    all_ops_list = {}
    schemas = torch._C._jit_get_all_schemas()

    for schema in schemas:
        schema_str = str(schema)
        if ".result" in schema.name or "!" in schema_str:
            continue
        if not schema.name.startswith("aten::"):
            continue
        if schema.name in ['aten::as_tensor', 'aten::tensor']:
            continue
        if schema.name.startswith("aten::_"):
            print(f"[Warning] Private API {schema.name} was skipped!")
            continue

        params = [arg.name for arg in schema.arguments]
        returns = [str(r.type) for r in schema.returns]
        if not returns or 'Tensor' not in returns[0]:  # not tensor operator API
            continue

        # get lower_conditions
        lower_conditions = []
        lower_func = extract_lowering_conditions.get_lowering_func(schema)
        if lower_func:
            configuration_slots = {arg.name: ("tensor" if "Tensor" in str(arg.type) else "attribute")
                                   for arg in schema.arguments}
            lower_conditions = extract_lowering_conditions.extract_conditions_from_fn(
                lower_func, configuration_slots=configuration_slots)
            print(f'[INFO] The conditions of {schema.name} is: ', lower_conditions)

        doc_str = None
        op_obj = my_obj.Operator(name=schema.name,
                                 parameters=params,
                                 returns=returns,
                                 signature=schema_str,
                                 doc=doc_str,
                                 lower_func = lower_func,
                                 lower_conditions=lower_conditions,)
        if schema.name not in all_ops_list.keys():
            all_ops_list[schema.name] = []
        all_ops_list[schema.name].append(op_obj)

    print(f"[INFO] Total aten api number is: {len(all_ops_list)}!")
    return all_ops_list


def generate_para_configs(op_name, sig, doc_str=''):
    """Dynamically generate valid inputs for each operator"""
    config_size = 5
    prompt_str = f"""Generate {config_size} diverse API parameter configurations for operator '{op_name}' strictly following requirements:

    1. **Parameter Types**
       - Tensor inputs: Must specify as [shape, dtype] tuples, such as ([1,2,3], 'float32')
       - Config parameters: Provide concrete values as Python dicts

    2. **Diversity Requirements**
       - Cover common and edge cases
       - Vary tensor dimensions (e.g., 2D, 3D, 4D)
       - Include different dtypes (e.g., 'bfloat16', 'float32', 'int64')
       - Use valid non-default parameter combinations

    3. **Output Format** 
       Strictly use Python dict containing each parameter configuration in the form: "param": value
       - Optional scalar/boolean parameters: use Python literals (True, False) with correct capitalization

    4. **Constraints**
       - Must use parameters defined in its signature:
       {sig}
       
       - Do NOT generate any invalid parameter values. 
       - None is not a valid value for any tensor argument.
       - Do NOT generate any parameter combinations that violate inter-parameter constraints.
       - No explanatory text

    Provide configurations as a Python list with {config_size} dict representing the configuration:"""

    spec_res_str = "["+call_llm(prompt_str)
    print(f"[INFO] The param value for {op_name}: {spec_res_str}\n")
    try:
        spec_res = eval(spec_res_str)
        assert isinstance(spec_res, list), "[ERROR] Generated all spec do not saved in the list!"
        for item in spec_res:
            assert isinstance(item, dict), f"[ERROR] The spec '{item}' do not saved in a dict!"
    except Exception as e:
        print(e)
        return []
    return spec_res


def generate_single_seed(op_name, para_config, seed_file_path='seed_tmp.py'):
    if "::" in op_name:
        op_name = op_name.split("::")[-1]

    all_params = list(para_config.keys())
    params_str = ", ".join(all_params)

    def _get_double_para_str(param_list):
        params_kv_str = param_list[0] + ", "
        if len(param_list) > 1:
            for para in param_list[1:]:
                params_kv_str += f"{para}={para},"  # for support the keyword-only parameter
        return params_kv_str

    param_kv_str = _get_double_para_str(all_params)

    op_func_str = f'''
def model_func({params_str}):
    out = torch.ops.aten.{op_name}({param_kv_str})
    return out

'''

    new_input_config = torch_utils.generate_api_inputs(para_config)
    op_input_str = "input_config = {"
    for k, v in new_input_config.items():
        if isinstance(v, str) and v in torch_utils.dtype_map.keys():
            op_input_str += f"'{k}':torch.{v},"
        # high-order tensor
        elif isinstance(v, list) and len(v)>=1 and isinstance(v[0], str) and v[0].startswith("torch.rand"):
            v_list_str = ', '.join(v)
            op_input_str += f"'{k}':[{v_list_str}],"
        elif isinstance(v, str) and torch_utils.is_true_str(v):
            op_input_str += f"'{k}':'{v}',"
        elif isinstance(v, str) and (v.startswith("cuda") or v.startswith("cpu")):
            op_input_str += f"'{k}':'{v}',"
        else:
            op_input_str += f"'{k}':{v},"
    op_input_str += "}"
    op_input_str = f'''try:
    {op_input_str}
except Exception as e:
    print(f"[Error]: Invalid operator Input.")
    sys.exit(0)
    '''

    seed_str_body = op_func_str + op_input_str
    seed_str = seed_str_header + seed_str_body + seed_str_foot
    with open(seed_file_path, "w") as f:
        f.write(seed_str)
    return seed_str





def single_op_test_gen(op_name, schema_id, op, seed_dir, op_info_4fusion):
    all_test_file_list = []

    para_config_list = generate_para_configs(op.name, op.signature)
    op.set_seeds(para_config_list)

    for test_id, para_config in enumerate(para_config_list):
        # [Warning] The same op_name for different schemas
        seed_test_file = os.path.join(seed_dir, f"seed_{op_name.split("::")[-1]}_{schema_id}_{test_id}.py")

        # Generate seed tests
        print(f"[INFO]: Para config for {op}: {para_config}")
        generate_single_seed(op_name, para_config, seed_test_file)
        all_test_file_list.append(seed_test_file)

        # mutate each seed based on condition
        op_lower_cond_list = op.lower_conditions if op.lower_conditions else []
        print(op_lower_cond_list, len(op_lower_cond_list))
        for cond_id, op_cond in enumerate(op_lower_cond_list):
            mutated_para_config_list = mutate_params.mutate_para_configs(op.name, op.signature, para_config, op_cond)
            for m_id, mutated_para_config in enumerate(mutated_para_config_list):
                mutated_test_file = os.path.join(seed_dir,
                                                 f"test_{op_name.split("::")[-1]}_{test_id}_M{cond_id}_{m_id}.py")
                generate_single_seed(op_name, mutated_para_config, mutated_test_file)
                all_test_file_list.append(mutated_test_file)

        # Fusion is opt-in; spawned workers inherit the environment.
        if os.getenv("LPTGEN_ENABLE_FUSION", "0") != "1":
            continue

        # mutate each seed based on op fusion
        assert op_name.startswith("aten:"), f"incorrect op name format (e.g., {op_name})"
        fusion_op_candidates = get_fusion_rules.get_fusion_candidates(op_name, op_info_4fusion)

        fusion_count = 3
        if len(fusion_op_candidates) < fusion_count:
            continue
        fusion_op_candidates = random.choices(fusion_op_candidates, k=fusion_count)
        for fusion_id, sub_op in enumerate(fusion_op_candidates):
            sub_op_name, sub_op_sig = sub_op
            mutated_test_str = mutate4opfusion.mutate_op_seq(seed_test_file, sub_op_name, sub_op_sig)
            if mutated_test_str == '':  # fail to mutate
                continue
            rand_id = random.randint(1, 1000)
            mutated_test_file = os.path.join(seed_dir,
                                             f"test_{op_name.split("::")[-1]}_{test_id}_MF{fusion_id}_{rand_id}.py")
            with open(mutated_test_file, "w") as f:
                f.write(mutated_test_str)
            all_test_file_list.append(mutated_test_file)
    return all_test_file_list


if __name__ == '__main__':
    op_name = "aten::_fused_dropout"
    para_config_list = [{"self": [[128, 256], "float32"], "p": 0.5, "generator": None},
                        {"self": [[10, 64, 128], "bfloat16"], "p": 0.3, "generator": None},
                        {"self": [[5, 8, 32, 64], "int64"], "p": 0.0, "generator": None},
                        {"self": [[[2, 3, 4], "bfloat16"], [[2, 3, 4], "bfloat16"]]},
                        {'self': [[4, 5], 'torch.float32'], 'dim': 0, 'index': [[3], 'int64'],
                         'source': [[3, 5], 'float32'], 'alpha': 2.5},
                        {'n': 8, 'dtype': 'torch.int64', 'layout': 'torch.strided', 'device': 'cuda:0', 'pin_memory': 'True', }
                        ]


    get_all_operators()
