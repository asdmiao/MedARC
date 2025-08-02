import os
import json
import numpy as np
import re
from rouge_score import rouge_scorer

# 配置选择模式：'highest'为最高置信度，'second_highest'为次高置信度
CONFIDENCE_SELECTION_MODE = 'second_highest'  # 可修改为 'highest' 或 'second_highest'

folder_path = "../MedARC/output"
file_path = os.path.join(folder_path, "factoid_ds_3_3.json")

# 打开文件并加载 JSON 数据
with open(file_path, "r", encoding='utf-8') as f:
    response_dict = json.load(f)

# 获取所有问题的列表
questions = list(response_dict.keys())
accuracies = []
rouge1_scores = []  
rouge2_scores = []      

# 初始化 ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

def clean_text(text):
    """清理文本，去除\, text, {, }等字符"""
    text = text.replace('\\', '').replace('text', '').replace('{', '').replace('}', '')
    return text.strip()

def extract_confidence(response):
    """从回答中提取置信度评分"""
    confidence_match = re.search(r'confidence\s*[:\-]?\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
    if confidence_match:
        try:
            return float(confidence_match.group(1))
        except ValueError:
            return 0.0
    return 0.0

# 遍历每个问题
for question in questions:
    responses, gt = response_dict[question]  # gt 是标准答案

    pred_solutions = []
    confidences = []
    
    for response in responses:
        assistant_responses = [r['content'] for r in response if r['role'] == 'assistant']
        for pred_solution in assistant_responses:
            # 提取置信度
            confidence = extract_confidence(pred_solution)
            confidences.append(confidence)
            
            # 提取boxed内容
            match = re.search(r'\\boxed{([^}]*)}', pred_solution)
            if match:
                cleaned_text = clean_text(match.group(1))
                pred_solutions.append(cleaned_text)
            else:
                pred_solutions.append("")  # 保持索引一致
    
    # 根据置信度选择预测答案
    selected_pred = None
    selected_confidence = 0.0
    
    if len(pred_solutions) >= 1:
        if len(pred_solutions) >= 2 and CONFIDENCE_SELECTION_MODE == 'second_highest':
            # 选择次高置信度的回答
            sorted_indices = np.argsort(confidences)[::-1]
            selected_idx = sorted_indices[0]
            selected_pred = pred_solutions[selected_idx]
            selected_confidence = confidences[selected_idx]
        else:
            # 默认选择最高置信度的回答
            selected_idx = np.argmax(confidences)
            selected_pred = pred_solutions[selected_idx]
            selected_confidence = confidences[selected_idx]
        
        print(f"Selected prediction for question {question} (confidence: {selected_confidence:.2f}): {selected_pred}")
        print(f"Ground truth: {clean_text(gt)}")
    else:
        print(f"Question {question} doesn't have any valid predictions")

    # 计算准确率
    cleaned_gt = clean_text(gt)
    if selected_pred is not None and selected_pred:  # 检查selected_pred不为None且非空
        acc = 1.0 if cleaned_gt.lower() in selected_pred.lower() else 0.0
    else:
        acc = 0.0

    accuracies.append(acc)

    # 计算ROUGE
    if selected_pred is not None and selected_pred:  # 检查selected_pred不为None且非空
        # 计算ROUGE分数
        scores = rouge_scorer_obj.score(cleaned_gt.lower(), selected_pred.lower())
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        
    else:
        rouge1 = rouge2 = 0.0

    rouge1_scores.append(rouge1)
    rouge2_scores.append(rouge2)

    # 输出当前问题的评分
    print(f"Question: {question}")
    print(f"  Selected prediction confidence: {selected_confidence:.2f}")
    print(f"  ROUGE-1: {rouge1:.4f}")
    print(f"  ROUGE-2: {rouge2:.4f}")
    print("-" * 50)

# 输出所有问题的平均评分
selection_mode_str = "highest" if CONFIDENCE_SELECTION_MODE != 'second_highest' else "second highest"
print(f"\nOverall Metrics (using {selection_mode_str} confidence predictions):")
print(f"  Accuracy: {np.mean(accuracies):.4f}")
print(f"  Average ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"  Average ROUGE-2: {np.mean(rouge2_scores):.4f}")