import json
import os
import re
os.environ['HF_ENDPOINT'] = ''
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def clean_text(text):
    """清理文本，去除\, text, {, }等字符"""
    text = text.replace('\\', '').replace('text', '').replace('{', '').replace('}', '')
    return text.strip()

def extract_final_answer(generated_text):
    """
    从生成的文本中提取最终答案（仅提取boxed内容，否则返回全文）
    :param generated_text: 生成的完整文本
    :return: 提取的答案文本
    """
    # 1. 只提取boxed内容
    boxed_match = re.search(r'\\boxed{([^}]*)}', generated_text)
    if boxed_match:
        return clean_text(boxed_match.group(1))
    
    # 2. 直接返回全文（去除首尾空白）
    return clean_text(generated_text)

def evaluate_results(generated_file_path, key="answer"):
    """
    综合评估函数（一次性计算所有指标）
    返回包含ROUGE和Cosine Similarity指标的字典
    
    :param generated_file_path: 生成结果的文件路径
    :param key: 标准答案的键名
    :return: 包含所有评估指标的字典
    """
    # 加载SBERT模型
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open(generated_file_path, 'r', encoding='utf-8', errors='replace') as f:
        generated_data = json.load(f)
    
    # 初始化评估工具
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], 
        use_stemmer=True
    )
    
    # 初始化结果存储
    results = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'cosine_similarity': []
    }
    
    total_count = len(generated_data)
    
    for question, (generated_text, true_answer) in generated_data.items():
        # 预处理答案
        pred_answer = extract_final_answer(generated_text).lower()
        true_answer = true_answer.strip().lower()
        
        
        # 计算ROUGE指标
        rouge_scores = scorer.score(true_answer, pred_answer)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            results[metric].append(rouge_scores[metric].fmeasure)
        
        
        # 计算SBERT + Cosine Similarity
        try:
            true_embedding = sbert_model.encode(true_answer, convert_to_tensor=True)
            pred_embedding = sbert_model.encode(pred_answer, convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(true_embedding, pred_embedding).item()
            results['cosine_similarity'].append(cosine_sim)
        except:
            results['cosine_similarity'].append(0)  # 如果计算失败，记为0
    
    # 计算最终结果
    for metric in ['rouge1', 'rouge2', 'rougeL', 'cosine_similarity']:
        results[metric] = sum(results[metric]) / total_count if total_count > 0 else 0
    
    return results

if __name__ == "__main__":
    # 文件路径配置
    result_file = '../MedARC/output/open-ended_0ds_1_1.json'
    
    # 执行评估
    evaluation = evaluate_results(result_file)
    
    # 美化输出
    print("\n=== Evaluation Results ===")
    print(f"ROUGE-1:            {evaluation['rouge1']:.4f}")
    print(f"ROUGE-2:            {evaluation['rouge2']:.4f}")
    print(f"ROUGE-L:            {evaluation['rougeL']:.4f}")
    print(f"Cosine Similarity:  {evaluation['cosine_similarity']:.4f}")
    print("="*25)