import json
import random

def extract_10_percent_to_jsonl(input_file_path, output_file_path):
    # 读取文件中的所有行
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    # 计算10%的行数
    num_lines = len(lines)
    num_to_extract = max(1, int(num_lines * 0.1))  # 至少提取1行
    
    # 随机选择行
    selected_lines = random.sample(lines, num_to_extract)
    
    # 将选中的行写入到新的.jsonl文件中
    with open(output_file_path, 'w') as output_file:
        for line in selected_lines:
            output_file.write(line)

# 使用示例
input_file_path = './data/raw_data.jsonl'  # 替换为你的.jsonl文件路径
output_file_path = './data/test.jsonl'  # 输出文件的路径
extract_10_percent_to_jsonl(input_file_path, output_file_path)