import os.path
import re
import sys


def find_max_iter(file_path):
    max_iteration = -1  # 初始化为 -1
    # 正则表达式匹配 d4rl_score 后面的数值
    iteration_pattern = re.compile(r'Iteration\s+(\d+)')
    current_iteration = -1

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                iteration_match = iteration_pattern.search(line)
                if iteration_match:
                    current_iteration = int(iteration_match.group(1))
                max_iteration = max(current_iteration, max_iteration)
        return max_iteration
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")
def find_max_d4rl_score(file_path):
    """
    读取指定文件，提取所有 d4rl_score 值，并返回最大值及其对应的 Iteration。

    :param file_path: 日志文件的路径
    :return: 最大的 d4rl_score 值及其对应的 Iteration
    """
    max_score = float('-inf')  # 初始化为负无穷
    max_iteration = -1  # 初始化为 -1
    # 正则表达式匹配 d4rl_score 后面的数值
    score_pattern = re.compile(r'd4rl_score:\s*([0-9.]+)')
    iteration_pattern = re.compile(r'Iteration\s+(\d+)')
    current_iteration = -1

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                iteration_match = iteration_pattern.search(line)
                if iteration_match:
                    current_iteration = int(iteration_match.group(1))
                score_match = score_pattern.search(line)
                if score_match:
                    score = float(score_match.group(1))
                    if score > max_score:
                        max_score = score
                        max_iteration = current_iteration
        return max_score, max_iteration
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误: {e}")


envs = ["halfcheetah", "hopper" , "walker2d"]
# envs = ["hopper"]
dataset_types = ["medium-expert", "medium", "medium-replay"]
score_list = []

# 使用参数进行传递
model_type = sys.argv[1]
dir_name = sys.argv[2]
dir_name= f"results/seed0/{dir_name}"
for dataset_type in dataset_types:
    for env in envs:
        file_name = f"{env}-{dataset_type}-{model_type}.log"
        file_name = os.path.join(dir_name, file_name)
        try:
            score, iteration = find_max_d4rl_score(file_name)
        except:
            continue
        score_list.append(score)
        max_iter = find_max_iter(file_name)
        print(f"{env}-{dataset_type}: {score} at Iteration {iteration} epochs:{max_iter}")
        # print(f"{env}-{dataset_type}: {score} at Iteration {iteration}")
# 打印均值
print(f"Mean: {sum(score_list) / len(score_list)}")