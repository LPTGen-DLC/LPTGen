import numpy as np
import random
from tvm import te, tir


def generate_te_placeholder(para_name, tensor_spec):
    _shape, _dtype = tensor_spec
    te_statement = f"te.placeholder({_shape}, dtype='{_dtype}', name='{para_name}')"
    return te_statement


def get_tvm_template(test_header_str, passes):
    target = 'llvm --num-cores=16'
    tvm_test_template = f'''import tvm
from tvm import te, topi, tir
from tvm import meta_schedule as ms
import numpy as np
import os, sys, shutil

try:
    from test_oracle import try_reference, assert_reference_outputs, NondeterministicReference
except ImportError:
    try_reference = None

def compile_mod(mod, np_input_list, op_outputs, opt_level=3):
    with tvm.transform.PassContext(opt_level):
        ref_mod = tvm.build(mod, target='llvm')
    mod_outputs = []
    for op_output in op_outputs:
        output_shape = op_output.shape
        output_type = op_output.dtype
        mod_outputs.append(tvm.nd.empty(output_shape, dtype=output_type, device=tvm.cpu(0)))

    tvm_inputs = [tvm.nd.array(x) for x in np_input_list]
    ref_mod(*tvm_inputs, *mod_outputs)
    return mod_outputs


{test_header_str}
python_reference = None
oracle_reason = "optional reference helper or operator metadata absent"
if use_python_reference and try_reference is not None:
    try:
        python_reference, oracle_reason = try_reference(
            globals().get('oracle_op_name'), input_config,
            globals().get('np_input_by_name', {{}}))
    except NondeterministicReference as exc:
        print(f"[ORACLE] Invalid test: {{exc}}")
        sys.exit(99)
    except Exception as exc:
        oracle_reason = f"reference preparation failed: {{exc}}"
print(f"[ORACLE] {{oracle_reason}}; mode={{'independent' if python_reference is not None else 'legacy'}}")
sch = tir.Schedule(te.create_prim_func(te_var_list_str+op_outputs).with_attr('target', tvm.target.Target('llvm')))
if not hasattr(op_outputs[0], "shape"):
    sys.exit(0)
ref_output = compile_mod(sch.mod, np_inputs, op_outputs, opt_level=0)
if python_reference is not None:
    assert_reference_outputs(python_reference, ref_output)

if os.path.exists('./tune_tmp'):
    shutil.rmtree('./tune_tmp')
database = ms.tir_integration.tune_tir(mod=sch.mod, target='{target}', work_dir='./tune_tmp', max_trials_global=1, num_trials_per_iter=1)
sch = ms.tir_integration.compile_tir(database, sch.mod, '{target}')
if not sch:
    sys.exit(0)
passes = [{','.join(passes)}]
with tvm.ir.transform.PassContext(opt_level=4):
    opt_mod = tvm.ir.transform.Sequential(passes)(sch.mod)
opt_mod_output = compile_mod(opt_mod, np_inputs, op_outputs, opt_level=4)

if python_reference is not None:
    assert_reference_outputs(python_reference, opt_mod_output)
else:
    for i in range(len(ref_output)):
        np.testing.assert_allclose(ref_output[i].numpy(), opt_mod_output[i].numpy(), rtol=1e-5, atol=1e-5,
        err_msg=f"An inconsistency detected.")
print("No bug found!")

'''
    return tvm_test_template


def generate_api_inputs(para_config):
    api_input_dict = {}
    new_config = {}
    te_placeholder_str = ''

    for name, para_val in para_config.items():
        if isinstance(para_val, (list, tuple)) and len(para_val) == 2 and isinstance(para_val[1], str):  # tensor spec
            api_input_dict[name] = f"np.random.uniform(-1, 1, size={para_val[0]}).astype('{para_val[1]}')"

            te_str = generate_te_placeholder(name, para_val)
            te_placeholder_str += f"    {name} = {te_str}\n"
            para_val = name
        new_config[name] = para_val
    print(te_placeholder_str, new_config)
    return te_placeholder_str, new_config, api_input_dict


def assign_schedule_params(sig, sch):
    blocks = sch.get_child_blocks(sch.get_block("root"))
    kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name == "block":
            if blocks:
                kwargs["block"] = blocks[0]
            else:
                return False
        elif param_name == "loop":
            if blocks and blocks[0]:
                loops = sch.get_loops(blocks[0])
                if loops:
                    kwargs["loop"] = loops[0]
        elif param_name == "new_order":
            if blocks and blocks[0]:
                block_rv = blocks[0]
                block = sch.get(block_rv)
                if hasattr(block, 'iter_vars'):
                    iter_vars = block.iter_vars
                    if iter_vars:
                        iter_var_indices = list(range(len(iter_vars)))
                        random.shuffle(iter_var_indices)
                        kwargs["new_order"] = iter_var_indices
        elif param_name.endswith("_index"):
            kwargs[param_name] = 0
        elif param_name == "factor":
            kwargs[param_name] = 2
        elif param_name == "preserve_unit_iters":
            kwargs[param_name] = True
        elif param_name in ["src_loop", "dst_loop"]:
            if blocks and blocks[0]:
                loops = sch.get_loops(blocks[0])
                if len(loops) >= 2:
                    kwargs[param_name] = loops[0 if param_name == "src_loop" else 1]
        else:
            if param.annotation == int:
                kwargs[param_name] = random.randint(-2, 5)
            elif param.annotation == bool:
                kwargs[param_name] = random.choice([True, False])
            elif param.annotation == str:
                kwargs[param_name] = ""
    return kwargs


def generate_transform_params(sig, sch):
    call_transform = ""
    return call_transform


def gen_test_header_str(op_name, te_placeholder_str, new_config, inputs_dict):
    new_config_str = ''
    for k, v in new_config.items():
        if k == v:
            new_config_str += f"'{k}': {v}, "
        elif isinstance(v, str):
            new_config_str += f"'{k}': '{v}', "
        else:
            new_config_str += f"'{k}': {v}, "
    new_config_str = 'input_config = {' + new_config_str + '}\n'
    np_input_str = 'np_inputs = [' + ','.join(inputs_dict.values()) + ']\n'
    te_var_name_list = str(list(inputs_dict.keys())).replace("'", "")
    te_var_list_str = f"te_var_list_str = {te_var_name_list}\n"

    all_params = new_config.keys()
    params_str = ','.join(all_params)
    param_kv_str = params_str

    model_func_str = f'''
def model_func({params_str}):
    out = {op_name}({param_kv_str})
    return out

'''
    # f"    op_outputs = {op_name}(**input_config)\n"
    test_header_str = model_func_str + f"\ntry:\n" \
                      f"{te_placeholder_str}" \
                      f"    {new_config_str}" \
                      f"    op_outputs = model_func(**input_config)\n"\
                      f"except Exception as e:\n" \
                      f"    sys.exit(99)\n" \
                      f"if not isinstance(op_outputs, (list, tuple)): \n" \
                      f"    op_outputs = [op_outputs]\n"

    test_header_str += np_input_str + te_var_list_str
    # Keep the same concrete values and parameter names for the optional reference.
    test_header_str += "use_python_reference = True\n"
    test_header_str += f"oracle_op_name = {op_name!r}\n"
    test_header_str += f"np_input_by_name = dict(zip({list(inputs_dict)!r}, np_inputs))\n"
    return test_header_str, model_func_str, f"{te_placeholder_str}\n{new_config_str}"


if __name__ == '__main__':
    pass
