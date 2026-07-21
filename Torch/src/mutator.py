import os
import random
import shutil

import triton
from utils import llm_utils
import re
import json
from irs import parse_triton_code


def extract_last_python_code(text):
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else text


class IRMutator:
    def __init__(self):
        pass

    def mutate_program(self, ir_program, target_pass, pattern):
        prompt = self._create_mutation_prompt(ir_program, target_pass, pattern)
        mutated_program = llm_utils.call_code_model(prompt)
        mutated_program = extract_last_python_code(mutated_program)
        return mutated_program

    def _create_mutation_prompt(self, ir_program, target_pass, pattern):
        prompt = f"""
        Your task is to mutate the given program so that it will trigger the `{target_pass}` optimization pass. 
        You should make changes to the program to meets the conditions for triggering this optimization.

        Conditions to trigger the optimization:
        {pattern}

        Original program:
        {ir_program}

        Important requirements:
        1. The mutated program must be syntactically and semantically valid, (e.g., forbid the use of non-existent APIs such as tl.shared_memory)
        2. The program must be directly runnable for compilation and inference
        3. No external function calls permitted in the mutated program  
        4. The program must be fully deterministic - avoid any random factors, undefined behavior, uninitialized memory, or non-deterministic operations

        Return ONLY the modified program without any additional explanation.
        """
        return prompt

    def validate_mutated_program(self, mutated_file, mutated_program_str, max_retries=5):
        retry_count = 0
        while retry_count < max_retries:
            # valid validation
            from utils import fuzz_utils
            ret_code, _, stderr = fuzz_utils.run_test(mutated_file)
            if ret_code:
                if "ast_to_ttir" in stderr or "'triton.language' has no attribute" in stderr:
                    print(f"[INVALID MUTATION] failed (attempt {retry_count + 1}):\n {stderr}")
                    retry_count += 1
                    if retry_count < max_retries:
                        mutated_program_str = self._fix_program(mutated_program_str, stderr)
                        parse_triton_code.replace_triton_in_seed_to_file(mutated_file, mutated_file, mutated_program_str)
                else:
                    return True, retry_count
            else:
                return True, retry_count
        return False, retry_count

    def _fix_program(self, program_str, error_msg):
        prompt = f"""
        The following IR program has syntax or semantic errors. Please fix the errors and ensure it is a valid, executable program that complies with the IR format.

        Error message:
        {error_msg}

        IR program with errors:
        {program_str}   

        Please provide the corrected IR program. The program must be syntactically and semantically valid. 
        Return ONLY the corrected code without any additional explanation.
        """
        fixed_program = llm_utils.call_code_model(prompt)
        fixed_program = extract_last_python_code(fixed_program)
        return fixed_program


def mutate_iteration(all_seed_list, pass_file, target_dir):
    def _select_pass(pass_list, select_num):
        selected_passes = random.choices(pass_list, k=select_num)
        return selected_passes

    all_mutations_list = []
    mutation_cnt = 0
    mutator = IRMutator()
    with open(pass_file, 'r', encoding='utf-8') as file:
        passes = json.load(file)

    for seed_file in all_seed_list:
        selected_pass_list = _select_pass(passes, 2)
        seed_str = parse_triton_code.extract_triton_from_seed(seed_file)
        for this_pass in selected_pass_list:
            pass_name = this_pass['pass_name']
            pass_pattern = this_pass['pattern']

            # mutate the seed_str on this_pass
            mutated_program = mutator.mutate_program(seed_str, pass_name, pass_pattern)
            print(mutated_program)
            target_file_name = f"{seed_file.split('/')[-1][:-3]}_{pass_name}_{mutation_cnt}.py"
            print(f"[DEBUG]: {target_file_name}")
            new_file_path = os.path.join(target_dir, target_file_name)

            mutate_flag = parse_triton_code.replace_triton_in_seed_to_file(seed_file, new_file_path, mutated_program)
            mutator.validate_mutated_program(new_file_path, mutated_program)

            if mutate_flag:
                all_mutations_list.append(new_file_path)
                mutation_cnt += 1
            if mutation_cnt > 10:
                return all_mutations_list
    return all_mutations_list


def test_single_mutate():
    # Testing
    file_path = "../results/triton_code_seed/888.py"
    pass_file = "/path/to/collect_pass/triton_passes.json"  # replace with your local pass file path

    mutator = IRMutator()
    seed_str = parse_triton_code.extract_triton_from_seed(file_path)

    with open(pass_file, 'r', encoding='utf-8') as file:
        passes = json.load(file)
    for item in passes:
        pass_name = item['pass_name']
        # pass_description = item['description']
        pass_pattern = item['pattern']

    mutated_program = mutator.mutate_program(seed_str, pass_name, pass_pattern)
    print(mutated_program)
    new_file_path = file_path.replace("triton_code_seed", "triton_code_mutation")
    parse_triton_code.replace_triton_in_seed_to_file(file_path, new_file_path, mutated_program)
    flag, times = mutator.validate_mutated_program(new_file_path, mutated_program)
    if not flag:
        os.remove(new_file_path)



if __name__ == "__main__":
    test_single_mutate()
