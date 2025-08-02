import os
import json
import numpy as np
import re
os.environ['HF_ENDPOINT'] = ''
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# 加载SBERT模型
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

folder_path = "../MedARC/output"
file_path = os.path.join(folder_path, "open-ended_ds_3_3.json")

# 打开文件并加载 JSON 数据
with open(file_path, "r", encoding='utf-8') as f:
    response_dict = json.load(f)

# 获取所有问题的列表
questions = list(response_dict.keys())
rouge1_scores = []  
rouge2_scores = []  
rouge_l_scores = []     
cosine_similarity_scores = []

# 初始化 ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

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
    
    # 选择置信度次高的预测答案
    selected_pred = None
    if len(pred_solutions) >= 2 and len(confidences) >= 2:  # 确保至少有2个回答
        # 获取置信度排序后的索引（从高到低）
        sorted_indices = np.argsort(confidences)[::-1]
        highest_idx = sorted_indices[0]  # 索引0是最高，1是次高
        selected_pred = pred_solutions[highest_idx]
        print(f"Selected prediction for question {question} (1nd highest confidence: {confidences[highest_idx]:.2f}): {selected_pred}")
        print(f"Ground truth: {clean_text(gt)}")
    elif len(pred_solutions) == 1:
        # 如果只有一个回答，就选择它（尽管它是最高也是最低）
        selected_pred = pred_solutions[0]
        print(f"Selected prediction for question {question} (only one response, confidence: {confidences[0]:.2f}): {selected_pred}")
        print(f"Ground truth: {clean_text(gt)}")
    else:
        print(f"Question {question} doesn't have any valid predictions")

    
    cleaned_gt = clean_text(gt)

    # 初始化所有评分指标
    rouge1 = rouge2 = rouge_l = bleu = cosine_sim = 0.0

    # 计算ROUGE和余弦相似度分数
    if selected_pred is not None and selected_pred:  # 检查selected_pred不为None且非空
        # 计算ROUGE分数
        scores = rouge_scorer_obj.score(cleaned_gt.lower(), selected_pred.lower())
        rouge1 = scores["rouge1"].fmeasure
        rouge2 = scores["rouge2"].fmeasure
        rouge_l = scores["rougeL"].fmeasure
        
        # 计算SBERT + Cosine Similarity
        gt_embedding = sbert_model.encode(cleaned_gt.lower(), convert_to_tensor=True)
        pred_embedding = sbert_model.encode(selected_pred.lower(), convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(gt_embedding, pred_embedding).item()

    rouge1_scores.append(rouge1)
    rouge2_scores.append(rouge2)
    rouge_l_scores.append(rouge_l)
    cosine_similarity_scores.append(cosine_sim)  # 存储余弦相似度

    # 输出当前问题的评分
    print(f"Question: {question}")
    if len(confidences) >= 2:
        print(f"  Selected prediction confidence (1nd highest): {np.sort(confidences)[-1]:.2f}")
    else:
        print(f"  Selected prediction confidence: {confidences[0] if confidences else 0.0:.2f}")
    print(f"  ROUGE-1: {rouge1:.4f}")
    print(f"  ROUGE-2: {rouge2:.4f}")
    print(f"  ROUGE-L: {rouge_l:.4f}")
    print(f"  Cosine Similarity: {cosine_sim:.4f}")
    print("-" * 50)

# 输出所有问题的平均评分
print(f"\nOverall Metrics (using 1nd highest confidence predictions):")
print(f"  Average ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"  Average ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"  Average ROUGE-L: {np.mean(rouge_l_scores):.4f}")
print(f"  Average Cosine Similarity: {np.mean(cosine_similarity_scores):.4f}")