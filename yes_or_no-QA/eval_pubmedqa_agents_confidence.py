import json
import os
import numpy as np
import re
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score


def parse_answer(input_str):
    """解析输入字符串为 'yes', 'no', 或 'maybe'"""
    input_str_lower = input_str.lower()
    if "yes" in input_str_lower:
        return "yes"
    elif "no" in input_str_lower:
        return "no"
    elif "maybe" in input_str_lower:
        return "maybe"
    else:
        return None


def most_frequent(List):
    """获取列表中出现次数最多的元素，并替换为列表最后一个元素"""
    if not List:
        return None

    frequency = {}
    for answer in List:
        if answer in frequency:
            frequency[answer] += 1
        else:
            frequency[answer] = 1

    # 找到出现次数最多的元素
    priority = {'yes': 3, 'no': 2, 'maybe': 1}
    max_count = 0
    best_answer = None

    for answer, count in frequency.items():
        if (count > max_count) or (count == max_count and priority[answer] > priority.get(best_answer, 0)):
            max_count = count
            best_answer = answer
    
    # 替换为列表最后一个元素
    return List[-1] if List else None


def extract_confidence(response):
    """从回答中提取置信度评分"""
    confidence_match = re.search(r'confidence\s*[:\-]?\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if confidence_match:
        try:
            return float(confidence_match.group(1))
        except ValueError:
            return 0.0
    return 0.0


if __name__ == "__main__":
    # 指定文件路径
    folder_path = "../MedARC/output"
    file_path = os.path.join(folder_path, "pubmedqa_ds_3_3.json")

    # 读取 JSON 数据
    with open(file_path, "r", encoding='utf-8') as f:
        response_dict = json.load(f)

    # 初始化真实值 (y_true) 和预测值 (y_pred)
    y_true = []
    y_pred = []

    # 遍历问题和预测结果
    questions = list(response_dict.keys())
    for question in questions[:]:
        responses, gt = response_dict[question]
        pred_solutions = []
        confidences = []
        
        # 提取预测解答和置信度
        for response in responses:
            assistant_responses = [r['content'] for r in response if r['role'] == 'assistant']
            for pred_solution in assistant_responses:
                # 提取置信度
                confidence = extract_confidence(pred_solution)
                confidences.append(confidence)
                
                # 提取boxed内容
                match = re.search(r"\\boxed{([^}]+)}", pred_solution)
                if match:
                    pred_solutions.append(match.group(1))
                else:
                    pred_solutions.append(assistant_responses[-1])
        
        # 根据置信度选择预测答案
        selected_pred = None
        if len(pred_solutions) >= 2 and len(confidences) >= 2:  # 确保至少有2个回答
            # 获取置信度排序后的索引（从高到低）
            sorted_indices = np.argsort(confidences)[::-1]
            second_highest_idx = sorted_indices[0]  # 索引0是最高，1是次高
            selected_pred = pred_solutions[second_highest_idx]
            print(f"Selected prediction for question {question} (2nd highest confidence: {confidences[second_highest_idx]:.2f}): {selected_pred}")
        elif len(pred_solutions) == 1:
            # 如果只有一个回答，就选择它
            selected_pred = pred_solutions[0]
            print(f"Selected prediction for question {question} (only one response, confidence: {confidences[0]:.2f}): {selected_pred}")
        else:
            print(f"Question {question} doesn't have any valid predictions")
            selected_pred = None

        # 计算最终预测
        final_prediction = parse_answer(selected_pred) if selected_pred else None

        # 将真实值和预测值加入列表
        y_true.append(parse_answer(gt))
        y_pred.append(final_prediction if final_prediction else "none")

    # 设置标签
    labels = ['yes', 'no', 'maybe']

    # 计算整体准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # 计算并打印精度、召回率、F1 值和分类报告
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    print("\nPrecision, Recall, F1 per class:")
    for i, label in enumerate(labels):
        print(f"Class: {label}, Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0,digits=3))