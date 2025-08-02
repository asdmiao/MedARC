import json 
import os
import numpy as np
import re
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def parse_answer(input_str):
    """将输入字符串解析为 'yes', 'no', 或 'maybe'"""
    if not isinstance(input_str, str):
        return None  # 确保输入是字符串
    input_str_lower = input_str.lower()
    
    if "yes" in input_str_lower:
        return "yes"
    elif "no" in input_str_lower:
        return "no"
    elif "maybe" in input_str_lower:
        return "maybe"

def compute_accuracy(gt, pred_solution):
    """计算准确性，比较 ground truth 和预测解答"""
    answers = gt.lower().strip()
    pred_answer = parse_answer(pred_solution) if pred_solution else None

    # 返回逻辑更新，判断字符串相等
    if pred_answer is None:
        return 0  # 如果没有预测答案，则认为错误
    elif answers == pred_answer:  # 直接比较字符串
        return 1
    else:
        return 0


if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "../MedARC/output"
    file_path = os.path.join(folder_path, "pubmedqa_0ds_1_1.json")

    # 打开文件并加载 JSON 数据
    with open(file_path, "r",encoding='utf-8') as f:
        response_dict = json.load(f)

    # 获取所有问题的列表
    questions = list(response_dict.keys())
    
    all_actual = []
    all_predicted = []

    # 遍历每个问题
    for question in questions:
        # responses 是包含多个对话回合的列表，gt 是 ground truth
        responses, gt = response_dict[question]
        pred_solutions = []

        # 提取所有 assistant 的回答内容
        for response in responses:
            if response['role'] == 'assistant':
                assistant_content = response['content']
                print(assistant_content)
                # 使用正则表达式提取 \\boxed{...} 中的内容
                match = re.search(r"\\boxed{(.*?)}", assistant_content)
                if match:
                    # 如果找到匹配项，则提取并添加到预测解答列表中
                    pred_solutions.append(match.group(1))
                else:
                    # 没有找到\\boxed，直接添加整个内容
                    pred_solutions.append(assistant_content)

        # 计算预测解答的准确性，并收集实际和预测答案
        for pred_solution in pred_solutions:
            # 将 ground truth 和预测结果加入到列表中
            actual_answer = parse_answer(gt)
            predicted_answer = parse_answer(pred_solution)
            all_actual.append(actual_answer)
            all_predicted.append(predicted_answer)

        # 输出准确性及相关信息
        print(f"Question: {question}")  
        print(f"Ground Truth: {gt}")
        print(f"Predicted Solutions: {pred_solutions}")


    # 计算并输出准确性
    accuracy = np.sum(np.array(all_actual) == np.array(all_predicted)) / len(all_actual)
    print("Overall Accuracy:", accuracy)

    # 计算并输出精度、召回率和 F1 值
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_actual, all_predicted, labels=['yes', 'no', 'maybe'], average=None
    )
    
    print("\nPrecision, Recall, F1 per class:")
    for label, p, r, f in zip(['yes', 'no', 'maybe'], precision, recall, f1):
        print(f"Class: {label}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

    # 总体评估报告
    print("\nClassification Report:")
    print(classification_report(all_actual, all_predicted, labels=['yes', 'no', 'maybe'], digits=3))