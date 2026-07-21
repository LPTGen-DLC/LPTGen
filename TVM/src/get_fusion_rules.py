import random


def get_fusion_candidates(before_op, all_operators, fusion_count):
    selected_candidates = random.choices(all_operators, k=fusion_count)
    return selected_candidates


if __name__ == '__main__':
    selected = get_fusion_candidates('', range(200), 5)
    print(selected)

