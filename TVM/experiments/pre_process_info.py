import os


def rename_file(info_f, new_info):
    print(info_f, new_info)
    with open(info_f, 'r') as f:
        all_content = f.read()
        all_content = all_content.replace('/path/to/whitefox/tvm/', '/software/tvm/')  # replace with your local TVM path mapping
        # all_content = all_content.replace('/path/to/tvm/', '/software/tvm/')  # replace with your local TVM path mapping if needed
    with open(new_info, 'w') as f:
        f.write(all_content)
    return


if __name__ == '__main__':
    ori_dir = '/share_container/optfuzz/res/cov_cumulate/whitefox'
    new_dir = '/share_container/optfuzz/res/cov_cumulate/final/whitefox'
    for file in os.listdir(ori_dir):
        ori_info = os.path.join(ori_dir, file)
        new_info = os.path.join(new_dir, file)
        rename_file(ori_info, new_info)
