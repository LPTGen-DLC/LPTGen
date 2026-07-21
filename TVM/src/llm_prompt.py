import inspect
from llm_utils import call_llm


def generate_para_configs(op_func):
    """Dynamically generate valid inputs for any operator"""
    op_name = op_func.__name__
    sig = inspect.signature(op_func)
    doc_str = inspect.getdoc(op_func)
    para_list = [p.name for p in sig.parameters.values()]
    print(para_list)
    # assert False

    config_size = 3
    prompt_str = f"""Generate {config_size} diverse API parameter configurations for {op_name} strictly following requirements:

    1. **Parameter Types**
       - Tensor inputs: Specify as [shape, dtype] tuples
       - Config parameters: Provide concrete values as Python dicts

    2. **Diversity Requirements**
       - Cover common and edge cases
       - Vary tensor dimensions (e.g., 2D, 3D)
       - Include different dtypes (float32, int64, etc)
       - Use valid but non-default parameter combinations

    3. **Output Format** 
       Strictly use Python dict containing each parameter configuration in the form: "param": value

    4. **Constraints**
       - ONLY use API parameters from documentation:
       {doc_str}
       - NO explanatory text
       - NO invalid parameter combinations

    Provide configurations as a Python list with {config_size} dict representing the configuration:"""

    spec_res_str = "["+call_llm(prompt_str)
    print(spec_res_str)
    try:
        spec_res = eval(spec_res_str)
        assert isinstance(spec_res, list), "[ERROR] Generated all spec do not saved in the list!"
        for item in spec_res:
            assert isinstance(item, dict), f"[ERROR] The spec '{item}' do not saved in a dict!"
    except Exception as e:
        print(e)
        return []
    return spec_res


if __name__ == '__main__':
    pass
