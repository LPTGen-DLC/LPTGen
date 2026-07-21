from llm_utils import call_llm
import inspect


def post_formal_for_llm(res_str):
    res_str = 'def model_func' + res_str.split('def model_func')[-1]
    res_str = res_str.split('[/')[0].strip()
    return_line = res_str.split('\n')[-1].strip()
    if 'return ' in res_str and not return_line.startswith('return '):
        res_str = res_str[:-len(return_line)]  # [RESPONSE END]
        return post_formal_for_llm(res_str)
    return res_str


def mutate_op_seq(seed_test_file, model_func_str, seed_input_config, sub_op_func):
    sub_op = sub_op_func.__name__
    sub_op_sig = inspect.signature(sub_op_func)
    sub_op_doc = inspect.getdoc(sub_op_func)

    with open(seed_test_file, 'r') as f:
        seed_str = f.read()
    mutated_func = mutate_fusion_llm(model_func_str, sub_op, sub_op_sig, seed_input_config, sub_op_doc)
    new_test = seed_str.replace(model_func_str, mutated_func)
    return new_test


def mutate_fusion_llm(model_func_str, sub_op, sub_op_signature, seed_input_config, sub_op_doc=''):
    prefix = model_func_str.split("return out")[0]
    prompt_str = f"""
You are given an existing Python function named `model_func`. 
Your task is to mutate this function by **inserting the operator call of `{sub_op}` ** into it.    
The signature of {sub_op} is: `{sub_op_signature}` 
It documentation is: {sub_op_doc}

Original function:
```
{model_func_str}
```

**Mutation Rules** (MANDATORY):
   - Keep the exactly same function name and the same parameters for the `model_func`.
   - Insert {sub_op} after the original computation, using previous output as its main input.
   - All arguments of `{sub_op}`, including optional ones, must be assigned valid values, do not rely on default values.
   - Avoid use any placeholder to assign the parameter of {sub_op}
   - The return value of `model_func` MUST be the output of `{sub_op}`.

**Validity Requirements** (STRICT):
- Every argument passed to {sub_op} MUST be: type-correct, shape-compatible (if tensor), semantically valid for the operator.
- The mutated function MUST be valid, executable Python code.
- The call to {sub_op} MUST be valid under the following input example: `{seed_input_config}`

**Output Format** (MANDATORY)
   - Must only output the **full mutated function**.
   - Do **not** include any explanations, comments

The full mutated function is:
```
{prefix}
"""
    print(prompt_str)
    res_str = call_llm(prompt_str, stop_str='return out```')

    # get clear result
    print(f"[DEBUG], Ori output: {res_str}")
    res_str = post_formal_for_llm(res_str)
    if len(res_str.split('\n')) <= 3:
        return ''
    print(f"[INFO] The mutated func is:\n{res_str}")
    return res_str


if __name__ == '__main__':
    seed_test_file = '/path/to/results/test_view_0_M0_0.py'  # replace with your local seed test path
    op_name = "topi.add"
    ori_model = """def model_func(x, y):
    out = torch.ops.aten.add.Tensor(x, y)
    return out
"""

    res = mutate_fusion_llm(ori_model, "torch.ops.aten.argwhere", 'aten::argwhere(Tensor self) -> Tensor', '', )

# res = mutate_fusion_llm_via_FIM(ori_model, "torch.ops.aten.argwhere", 'aten::argwhere(Tensor self) -> Tensor','',)
