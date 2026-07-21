import datetime
import time
import os
import random
import shutil
import subprocess
from tvm.tir.schedule import Schedule

import inspect
from typing import Callable
from llm_utils import call_llm
import tvm_utils
import collects_apis
import multiprocessing
from multiprocessing import Value, Lock


import get_fusion_rules
import mutate4opfusion


def collect_cov(cov_cnt):
    print(f"[INFO] Collect the coverage times: {cov_cnt}.")
    sut_cov_dir = "/path/to/tvm/build/CMakeFiles/tvm_objs.dir/src/"  # replace with your local TVM build directory
    lcov_bin = "/path/to/lcov"  # replace with your local lcov executable
    cov_collect = (
        f"{lcov_bin} --capture --directory {sut_cov_dir} "
        f"--output-file ./results_fuzz/LPTGen_{cov_cnt}.info "
        f"--rc lcov_branch_coverage=1"
    )
    subprocess.run(cov_collect, shell=True)


def cov_init():
    srcpaths = ["/path/to/tvm/build/CMakeFiles/tvm_objs.dir/src/"]  # replace with your local TVM build directory
    for gcdapath in srcpaths:
        cmd = 'find {} -name *.gcda -type f | xargs rm -rf'.format(gcdapath)
        print(f"[INFO] Remove all history coverage by command: {cmd}")
        status = os.system(cmd)
        if status:
            raise Exception('Fail in deleting gcda files')

# -----------------------------
# Global variable
class G:
    test_num = 0
    succeed_num = 0
    bugs_num = Value('i', 0)
    bugs_num_lock = Lock()

    invalid_num = Value('i', 0)
    invalid_num_lock = Lock()

    valid_num = Value('i', 0)
    valid_num_lock = Lock()


all_schedules = collects_apis.get_all_tir_schedules()


def check_para_validity(param_config_list_str, true_param_list, op_name):
    try:
        params_config_list = eval(param_config_list_str)
        assert isinstance(params_config_list, (list, tuple)), "[ERROR] Generated all spec do not saved in a list!"
    except Exception as e:
        print(e)
        return []
    processed_params_config_list = []
    for gen_param_dict in params_config_list:
        if not isinstance(gen_param_dict, dict):
            print(f"[ERROR] The spec '{gen_param_dict}' do not saved in a dict!")
            continue
        if sorted(list(gen_param_dict.keys())) != sorted(true_param_list):
            with open('doc_bug.txt', 'a') as f:
                f.write(f"{op_name}: docstring para:{gen_param_dict.keys()}, true para: {true_param_list}\n\n")

        gen_param_dict = {k: v for k, v in gen_param_dict.items() if k in true_param_list}
        if sorted(list(gen_param_dict.keys())) == sorted(true_param_list):  # correct or fixed
            processed_params_config_list.append(gen_param_dict)
        elif set(gen_param_dict.keys()) - set(true_param_list) == set():
            processed_params_config_list.append(gen_param_dict)  # just lack some params (maybe is the optional params)
        else:
            continue
    return processed_params_config_list


# ------------------------------
# Module: Smart Input Generation
# ------------------------------
def generate_para_configs(op_func: Callable, config_size):
    """Dynamically generate valid inputs for any operator"""
    op_name = op_func.__name__
    sig = inspect.signature(op_func)
    doc_str = inspect.getdoc(op_func)

    true_param_list = [p.name for p in sig.parameters.values()]
    if 'Parameters' not in doc_str:
        doc_str += f"\nParameters are: {true_param_list}\n"
    # print(true_param_list)

    prompt_str = f"""Generate {config_size} diverse API parameter configurations for {op_name} strictly following requirements:

    1. **Parameter Types**
       - Tensor inputs: Specify as [shape, dtype] tuples
       - Config parameters: Provide concrete values as Python dicts

    2. **Diversity Requirements**
       - Cover common and edge cases
       - Vary tensor dimensions (e.g., 2D, 3D)
       - Include different dtypes (float32, int16, etc)
       - Use valid but non-default parameter combinations

    3. **Output Format** 
       Strictly use Python dict containing each parameter configuration in the form: "param": value

    4. **Constraints**
       - ONLY use API parameters from documentation:
       {doc_str}
       - NO explanatory text
       - NO invalid parameter combinations

    Provide configurations as a Python list with {config_size} dict representing the configuration:"""

    param_config_list_str = "[" + call_llm(prompt_str)
    print(param_config_list_str)

    params_config_list = check_para_validity(param_config_list_str, true_param_list, op_name)
    return params_config_list


# ------------------------------
# Module: Safe schedule
# ------------------------------
def apply_safe_schedule(
        sch: Schedule,
        pass_name: str,
) -> bool:
    """Safely apply a schedule with proper error handling"""
    try:
        method = getattr(sch, pass_name.split(".")[1])
        sig = inspect.signature(method)
        doc_str = inspect.getdoc(method)
        kwargs = tvm_utils.assign_schedule_params(sig, sch)
        method(**kwargs)
    except Exception as e:
        print(f"[DEBUG]: Failed to apply {pass_name}: {str(e)}")
        return False
    return True


def apply_safe_transform(mod):
    try:
        mod = collects_apis.select_transforms(mod)
    except Exception as e:
        print(f"[DEBUG]: Failed to apply the tir.transform: {str(e)}")
    return mod


# ------------------------------
# Module: Generate test and Execute Differential Testing
# ------------------------------
def fuzz_single_api(op_name: str, op_func: Callable, all_operators, max_schedules: int = 3):
    para_config_list = generate_para_configs(op_func, 5)
    # para_config_list = [
    #     {'data': [(1, 3, 224, 224), "float32"], 'output_size': (112, 112), 'pool_type': 'max', 'layout': 'NCHW'}]
    for test_id, para_config in enumerate(para_config_list):
        te_placeholder_str, new_config, inputs_dict = tvm_utils.generate_api_inputs(para_config)
        print(f'[INFO] generated config for the {op_name}:', new_config)

        test_header_str, model_func_str, input_config = tvm_utils.gen_test_header_str(op_name, te_placeholder_str, new_config, inputs_dict)
        # applied_passes = []
        # for _ in range(max_schedules):
        #     pass_name, pass_func = random.choice(all_schedules)
        #     if apply_safe_schedule(sch, pass_name):
        #         applied_passes.append(pass_name)
        # mod = apply_safe_transform(sch.mod)

        passes = collects_apis.select_transforms()
        full_test_str = tvm_utils.get_tvm_template(test_header_str, passes)
        # print(full_test_str)

        new_test_file_name = f"{op_name}_{test_id}.py"
        new_test_path = os.path.join(test_dir, new_test_file_name)
        with open(new_test_path, 'w') as f:
            f.write(full_test_str)
        execute_test(new_test_path)  # Execute the test

        # Mutate for operator fusion
        seed_test_file = new_test_path
        fusion_count = 3
        fusion_op_candidates = get_fusion_rules.get_fusion_candidates(op_name, all_operators, fusion_count)
        for fusion_id, sub_op in enumerate(fusion_op_candidates):
            sub_op_name, sub_op_func = sub_op
            mutated_test_str = mutate4opfusion.mutate_op_seq(seed_test_file, model_func_str, input_config, sub_op_func)
            if mutated_test_str == '':  # fail to mutate
                continue
            rand_id = random.randint(1, 10000)
            mutated_test_file = os.path.join(test_dir, f"{op_name}_{test_id}_MF{fusion_id}_{rand_id}.py")
            with open(mutated_test_file, "w") as f:
                f.write(mutated_test_str)
            execute_test(mutated_test_file)  # Execute the test


def execute_test(new_test_path):
    flag = subprocess.run([f"python", new_test_path], capture_output=True, text=True)
    if flag.returncode == 99:
        # [Invalid Test]
        with G.invalid_num_lock:
            G.invalid_num.value += 1
        print(f"❗Script {new_test_path} failed with Invalid tests")
        return False, flag.stderr
    if flag.returncode != 0:
        # filter out false positives
        if "CUDA error: out of memory" in flag.stderr or "Unsupported" in flag.stderr or "NotImplementedError" in flag.stderr:
            return False, str(flag.stderr)
        print(f"❌ Script {new_test_path} failed with exit code {flag.returncode}.")
        print(f"Error output:\n{flag.stderr}")
        print(f"{new_test_path}: {flag.stderr}")
        shutil.copy(new_test_path, bugs_dir)
        with G.bugs_num_lock:
            G.bugs_num.value += 1
        return True, flag.stderr
    else:
        with G.valid_num_lock:
            G.valid_num.value += 1
        print(f"✅ [Successfully]: {new_test_path}")
        return True, ""


def worker_wrapper(args):
    op_name, op_func, all_operators = args
    try:
        result = fuzz_single_api(op_name, op_func, all_operators)
        return op_name, True, result
    except Exception as e:
        error_msg = f"Operation {op_name} crashed:\n{e}"
        return op_name, False, error_msg


def multi_process_fuzz(all_tasks, num_processes):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count() // 2
    bug_count = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker_wrapper, all_tasks)
    with open("bug.txt", "w") as bug_file:
        for op_name, success, result in results:
            if not success:
                bug_count += 1
                bug_file.write(f"=== Bug found in {op_name} ===\n")
                bug_file.write(result)
                bug_file.write("\n\n")
    print(f"Fuzzing completed. Found {bug_count} bugs. See bug.txt for details.")


# ------------------------------
# Module: Main Test Runner
# ------------------------------
def main(namespace_list):
    total_test_num = 0
    fuzz_start_time = time.time()
    last_save_cov_time = fuzz_start_time
    fuzz_execution_time = 2000*24
    cov_during_time = 1800
    cov_cnt = 0

    # cov_init()  # delete all history cov info
    cov_cnt = 18  # continue

    all_operators = []
    for namespace in namespace_list:
        operators = collects_apis.get_topi_operators(namespace)
        all_operators.extend(operators)
    print(f"[INFO] Discovered {len(all_operators)} API!")
    all_tasks = []

    restart_index = 0

    for i, (op_name, op_func) in enumerate(all_operators):
        all_tasks.append([op_name, op_func, all_operators])
        # print(op_name)

        # if 'topi.get_const_tuple' in op_name:
        #     restart_index = i + 1
        # fuzz_single_api(op_name, op_func, all_operators)
    # assert False, f'restart from the id: {restart_index}'

    # Fuzz batch_size lowering APIs at once!
    batch_size = 6
    for i in range(restart_index, len(all_tasks), batch_size):
        batch_tasks = all_tasks[i:i + batch_size]
        multi_process_fuzz(batch_tasks, batch_size)

        current_time = time.time()
        during_time = current_time - last_save_cov_time
        if during_time > cov_during_time:
            collect_cov(cov_cnt)
            cov_cnt += 1
            last_save_cov_time = time.time()
            cov_collected_time = last_save_cov_time - current_time
            fuzz_execution_time += cov_collected_time  # fuzzing time exclude the coverage collection time

        total_fuzz_time = current_time - fuzz_start_time
        if total_fuzz_time > fuzz_execution_time:
            print(f"[INFO]: Total generated tests number is: {total_test_num};")
            print("[INFO]: Finished ALL && Timeout!")
            return True
    print("[INFO] FINISH ALL!")


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    test_dir = f"./results_fuzz/{current_time}/tests"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        shutil.copy("./collects_apis.py", test_dir)
    bugs_dir = f"./results_fuzz/{current_time}/failure"
    if not os.path.exists(bugs_dir):
        os.makedirs(bugs_dir)
        shutil.copy("./collects_apis.py", test_dir)

    namespace_list = ["topi", "topi.nn", "topi.image", "topi.cpp"]  # "topi.gpu"
    main(namespace_list)

    '''
    python fuzz.py
    '''
