import json
import re
from rouge_score import rouge_scorer

def extract_final_answer(generated_text):
    """
    从生成的文本中提取最终答案（仅提取boxed内容，否则返回全文）
    :param generated_text: 生成的完整文本
    :return: 提取的答案文本
    """
    # 1. 只提取boxed内容
    boxed_match = re.search(r'\\boxed{([^}]*)}', generated_text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 2. 直接返回全文（去除首尾空白）
    return generated_text.strip()

def evaluate_results(generated_file_path, key="answer"):
    """
    综合评估函数（一次性计算所有指标）
    返回包含accuracy、ROUGE指标的字典
    
    :param generated_file_path: 生成结果的文件路径
    :param key: 标准答案的键名
    :return: 包含所有评估指标的字典
    """
    with open(generated_file_path, 'r', encoding='utf-8', errors='replace') as f:
        generated_data = json.load(f)
    
    # 初始化评估工具
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2'], 
        use_stemmer=True
    )
    
    # 初始化结果存储
    results = {
        'accuracy': 0,
        'rouge1': [],
        'rouge2': []
    }
    
    correct_count = 0
    total_count = len(generated_data)
    
    for question, (generated_text, true_answer) in generated_data.items():
        # 预处理答案
        pred_answer = extract_final_answer(generated_text).lower().strip()
        true_answer = true_answer.strip().lower()
        
        # 计算准确率
        if true_answer == pred_answer or true_answer in pred_answer:
            correct_count += 1
        
        # 计算ROUGE指标
        rouge_scores = scorer.score(true_answer, pred_answer)
        for metric in ['rouge1', 'rouge2']:
            results[metric].append(rouge_scores[metric].fmeasure)
        
    
    # 计算最终结果
    results['accuracy'] = correct_count / total_count if total_count > 0 else 0
    for metric in ['rouge1', 'rouge2']:
        results[metric] = sum(results[metric]) / total_count if total_count > 0 else 0
    
    return results

if __name__ == "__main__":
    # 文件路径配置
    result_file = '../MedARC/output/factoid_0ds_1_1.json'
    
    # 执行评估
    evaluation = evaluate_results(result_file)
    
    # 美化输出
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"ROUGE-1:  {evaluation['rouge1']:.4f}")
    print(f"ROUGE-2:  {evaluation['rouge2']:.4f}")
    print("="*25)