import datetime
import seed_generator

import os
import sys
import time
import random
from datetime import datetime
import shutil
import multiprocessing
import subprocess
from utils import fuzz_utils
import get_fusion_rules

env = os.environ.copy()
env["GCOV_FLUSH_INTERVAL"] = "1"




def collect_cov(cov_cnt):
    print(f"[INFO] Collect the coverage times: {cov_cnt}.")
    sut_cov_dir = "/path/to/pytorch/build/"  # replace with your local PyTorch build directory
    lcov_bin = "/path/to/lcov"  # replace with your local lcov executable
    cov_collect = (
        f"{lcov_bin} --capture --directory {sut_cov_dir} --no-external "
        f"--output-file ../results/LPTGen_{cov_cnt}.info "
        f"--rc lcov_branch_coverage=1"
    )
    subprocess.run(cov_collect, shell=True)


def cov_init():
    srcpaths = ["/path/to/pytorch/source/"]  # replace with your local PyTorch source directory
    for gcdapath in srcpaths:
        cmd = 'find {} -name *.gcda -type f | xargs rm -rf'.format(gcdapath)
        print(f"[INFO] Remove all history coverage by command: {cmd}")
        status = os.system(cmd)
        if status:
            raise Exception('Fail in deleting gcda files')


class Fuzzer:
    def __init__(self, _execution_time, _res_dir):
        self.res_dir = _res_dir
        self.log_file = f"{_res_dir}/res_fuzzer_log.txt"
        self.failure_dir = f"{_res_dir}/res_failures"
        self.execution_dir = f"{_res_dir}/res_executions"
        self.start_time = time.time()
        self.execution_time = fuzz_utils.parse_execution_time(_execution_time)
        os.makedirs(self.failure_dir, exist_ok=True)
        os.makedirs(self.execution_dir, exist_ok=True)

        self.manager = multiprocessing.Manager()
        self.invalid_mutant_num = multiprocessing.Value('i', 0)
        self.all_detected_unique_bugs = self.manager.list()
        self.performance_bugs_num = multiprocessing.Value('i', 0)

    def log_bug(self, test_case_path, stderr):
        with open(self.log_file, 'a') as f:
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Test Case Path: {test_case_path}\n")
            f.write(f"ERROR: {stderr}\n")
            f.write("=" * 66 + "\n")
        shutil.copy(test_case_path, self.failure_dir)

    def single_task(self, total_test_num, seed_test_file):

        ret_code, _, stderr = fuzz_utils.run_test(seed_test_file)
        if ret_code:  # detected a bug
            if "not implemented" in stderr or "support" in stderr or "NotImplementedError" in stderr:
                print(f"[WARNING] SKip the unsupported issue\n{stderr}\n")
                return
            elif "CUDA error: out of memory" in stderr:
                print(f"[WARNING]: cuda out of memory!\n")
                return
            elif "must be " in stderr or "must have" in stderr:
                print(f"[WARNING]: Invalid input value\n{stderr}\n")
                return
            stderr = fuzz_utils.simple_crash_message(stderr)
            unique_bug_mess = fuzz_utils.extract_unique_crash_message(stderr)
            if unique_bug_mess not in self.all_detected_unique_bugs or "Segmentation fault" in unique_bug_mess:
                print(unique_bug_mess)
                self.all_detected_unique_bugs.append(unique_bug_mess)
                self.log_bug(seed_test_file, stderr)
                print(f"[INFO] Detected bug number: {len(self.all_detected_unique_bugs)}")
        else:
            print(f"[SUCCESS]: No bug detected using the test: {seed_test_file}!")
        return

    def fuzz(self):
        total_test_num = 0
        last_save_cov_time = time.time()
        cov_during_time = 1800
        cov_cnt = 0
        cov_init()  # delete all history cov info

        while True:
            # #####################################[Gen Tests]####################################
            all_tests_file_list = []

            all_ops = seed_generator.get_all_operators()
            items = list(all_ops.items())
            random.shuffle(items)
            all_ops = dict(items)

            op_info_4fusion = get_fusion_rules.build_op_info()

            total_test_count = 0
            for op_name, op_list in all_ops.items():
                for schema_id, op in enumerate(op_list):  # each op have several different schemas
                    print(f"[INFO] Test {op.signature}")
                    op_test_files_list = seed_generator.single_op_test_gen(op_name, schema_id, op, self.execution_dir, op_info_4fusion)
                    all_tests_file_list.extend(op_test_files_list)
                    total_test_count += len(op_test_files_list)

                    # ####################################[Run Tests]####################################
                    processes = []
                    for test_file_path in op_test_files_list:
                        total_test_num += 1
                        p = multiprocessing.Process(target=self.single_task, args=(total_test_num, test_file_path,))
                        processes.append(p)
                        p.start()

                        if len(processes) >= 32:
                            processes[0].join()
                            processes.pop(0)
                    for p in processes:
                        p.join()

                    print(f"[INFO] Finish run tests for the operator {op_name}!")
                    current_time = time.time()
                    during_time = current_time - last_save_cov_time
                    if during_time > cov_during_time:
                        collect_cov(cov_cnt)
                        cov_cnt += 1
                        last_save_cov_time = time.time()
                        cov_collected_time = last_save_cov_time - current_time
                        self.execution_time += cov_collected_time  # fuzzing time exclude the coverage collection time

                    total_fuzz_time = current_time - self.start_time
                    if total_fuzz_time > self.execution_time:
                        print(f"[INFO]: Total generated tests number is: {total_test_num};"
                              f"Invalid tests number is:{self.invalid_mutant_num.value}; "
                              f"Valid rate is: {1 - self.invalid_mutant_num.value / total_test_num}")
                        print(f"[INFO]: Total detected unique bugs number is:{len(self.all_detected_unique_bugs)}")
                        print(f"[INFO]: Total detected performance bugs is:{self.performance_bugs_num.value}")
                        print("[INFO]: Finished ALL && Timeout!")
                        return True


if __name__ == "__main__":
    res_dir = '../results'
    execution_time = sys.argv[1] if len(sys.argv) > 1 else "13h"

    res_dir = os.path.join(res_dir, f"{datetime.now().strftime("%m-%d_%H-%M")}")
    if not os.path.exists(res_dir):
        print(res_dir)
        os.makedirs(res_dir)

    fuzzer = Fuzzer(execution_time,res_dir)
    fuzzer.fuzz()
    '''
    python run_fuzz.py 1h
    '''
