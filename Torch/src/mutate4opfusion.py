from sys import prefix

from utils.llm_utils import call_FIM_llm, call_llm
import re


def post_formal_for_llm(res_str):
    res_str = 'def model_func' + res_str.split('def model_func')[-1]
    res_str = res_str.split('[/')[0].strip()
    return_line = res_str.split('\n')[-1].strip()
    if 'return ' in res_str and not return_line.startswith('return '):
        res_str = res_str[:-len(return_line)]  # [RESPONSE END]
        return post_formal_for_llm(res_str)
    return res_str


def extract_op_config_line(code_str):
    match = re.search(r"^\s*(input_config\s*=\s*\{.*\})", code_str, re.MULTILINE)
    if not match:
        print(f"[ERROR] input_config line not found")
        return ''
    return match.group(1).strip()


def extract_model_func(seed_str):
    pattern = re.compile(r"(def\s+model_func\s*\(.*?\)\s*:\s*(?:.|\n)*?^\s*return\s+out\s*$)", re.MULTILINE)
    match = pattern.search(seed_str)
    if not match:
        print("[ERROR] model_func definition with 'return out' not found!")
        return ''
    model_func_str = match.group(1)
    return model_func_str

def mutate_op_seq(seed_test_file, sub_op, sub_op_signature):
    with open(seed_test_file, 'r') as f:
        seed_str = f.read()

    model_func_str = extract_model_func(seed_str)
    seed_input_config = extract_op_config_line(seed_str)  # 'input_config=...'
    if model_func_str == '' or seed_input_config == '':
        print("[ERROR] input_config not found!")
        return ''

    mutated_func = mutate_fusion_llm(model_func_str, sub_op, sub_op_signature, seed_input_config)
    new_test = seed_str.replace(model_func_str, mutated_func)
    return new_test


def mutate_fusion_llm(model_func_str, sub_op, sub_op_signature, seed_input_config):
    prefix = model_func_str.split("return out")[0]

    prompt_str = f"""
    You are given an existing Python function named `model_func`. 
    Your task is to mutate this function by **inserting the operator call of `{sub_op}` ** into it.    
    The signature of {sub_op} is: `{sub_op_signature}` 

    Original function:
    ```
    {model_func_str}
    ```
    
    **Mutation Rules** (MANDATORY):
       - Keep the exactly same function name and the same parameters for the `model_func`.
       - Insert {sub_op} after the original computation, using previous output as its main input.
       - All other arguments of `{sub_op}`, including optional ones, must be assigned valid values, do not rely on default values.
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
    print(f"[DEBUG] LLM Output: {res_str}")

    # get clear result
    res_str = post_formal_for_llm(res_str)
    print(f"[INFO] The mutated func is:\n{res_str}")
    return res_str

if __name__ == '__main__':
    seed_test_file = '/path/to/results/test_view_0_M0_0.py'  # replace with your local seed test path
    op_name = "aten::_fused_dropout"
    ori_model = """def model_func(x, y):
    out = torch.ops.aten.add.Tensor(x, y)
    return out
"""

    res = mutate_fusion_llm(ori_model, "torch.ops.aten.argwhere", 'aten::argwhere(Tensor self) -> Tensor','',)

    # res = mutate_op_seq(seed_test_file, "torch.ops.aten.argwhere", "aten::argwhere(Tensor self) -> Tensor", '')
    # print(res)

#     res = '''PYTHON]
# def model_func(grad_output, self):
#     out = torch.ops.aten.hardswish_backward(grad_output, self=self,)
#     out = torch.ops.aten.fix.out(out, out=torch.empty_like(out))
#     return out
# [/PYTHON]
#
#     '''
#     res = post_formal_for_llm(res)
#     print(res)

    # res = mutate_fusion_llm_via_FIM(ori_model, "torch.ops.aten.argwhere", 'aten::argwhere(Tensor self) -> Tensor','',)
