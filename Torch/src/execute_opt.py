import random
import json
from collect_pass import pass_clazz
from mutator import IRMutator
from collect_pass import parse_all_passes_from_td


def load_passes(pass_file):
    with open(pass_file, 'r', encoding='utf-8') as file:
        passes_json = json.load(file)
    return passes_json


def select_passes(passes_list):
    passes_json = random.sample(passes_list, 3)
    return passes_json


def run_pass(seed, this_pass: pass_clazz.TritonPass, new_test_path, pass_cnt):
    pass_name = this_pass.pass_name
    pass_description = this_pass.description

    mutator = IRMutator()
    mutated_program = mutator.mutate_program(seed, pass_name, pass_description)
    if mutated_program == seed:
        print("[Warning] The mutated generated the same program as the seed~!")
        return False

    valid_flag, retry_count = mutator.validate_mutated_program(seed, max_retries=2)
    if valid_flag:
        print(f"[INFO] Mutated tests are valid with try time: {retry_count}")
    else:
        print(f"[Warning] Mutated tests are Invalid exceeding the max retry time: {retry_count}")
        return False

    new_mutated_test_path = new_test_path[:-3] + f"_M{pass_cnt}.py"

    with open(new_mutated_test_path, 'w') as f:
        f.write(mutated_program)
    return new_mutated_test_path


def run_mutate_iterate():
    pass_list = parse_all_passes_from_td.load_triton_passes("./collect_pass/triton_passes.json")
    for pazz in pass_list:
        print(pazz.pass_name)




if __name__ == '__main__':
    run_mutate_iterate()

