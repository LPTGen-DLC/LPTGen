from llm_utils import call_llm


def mutate_para_configs(op_name, sig, seed_config, low_cond, doc_str='', config_size=3):
    """Dynamically mutate to generate valid inputs for each operator"""

    prompt_str = f"""
Generate {config_size} API parameter configuration for operator '{op_name}' by mutating the seed configuration.
The seed configuration is {seed_config}
The configuration mutation should try to meet the condition of: {low_cond}.

The mutated configuration must still strictly meets the following requirements:
1. **Parameter Types**
   - Tensor inputs: Must specify as [shape, dtype] tuples, such as ([1,2,3], 'float32')
   - Config parameters: Provide concrete values as Python dicts

2. **Constraints**
   - Must use parameters defined in its signature:
   {sig}

   - Do NOT generate any invalid parameter values. 
   - None is not a valid value for any tensor argument.
   - Do NOT generate any parameter combinations that violate inter-parameter constraints.
   - No explanatory text

3. **Output Format** 
   Strictly use Python dict containing each parameter configuration in the form: "param": value
   - Optional scalar/boolean parameters: use Python literals (True, False) with correct capitalization
Provide configurations as a Python list with {config_size} dict representing the configuration:"""

    spec_res_str = "[" + call_llm(prompt_str)
    print(f"[INFO] param value for{op_name}: {spec_res_str}\n")
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
    op_name = "topi.add"
    para_config_list = []

    res = mutate_para_configs(op_name, '', para_config_list[1], 'V.graph.layout_opt and ndim == 2')
    print(res)
