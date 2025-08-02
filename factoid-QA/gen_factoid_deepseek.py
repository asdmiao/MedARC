import time
import random
import json
import os
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url=""
)


def generate_summary(responses):
    if not responses:
        return "No other opinions available."
    
    content = "Provide a clear summary of key similarities and differences in these responses. Keep it concise."
    for i, response in enumerate(responses, 1):
        content += f"Response {i}: {response}\n"
    
    messages = [
        {"role": "system", "content": "You are an expert at identifying consensus and disagreement."},
        {"role": "user", "content": content}
    ]
    
    api_call = lambda: client.chat.completions.create(
        model="deepseek-v3",
        messages=messages,
        n=1
    )
    
    completion = call_api_with_retry(api_call)
    return completion.choices[0].message.content

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, the answer should be a concise sentence or short phrase."}

    other_responses = []
    for agent in agents:
        if len(agent) > idx:
            agent_response = agent[idx]["content"]
            other_responses.append(agent_response)

    summary = generate_summary(other_responses)
    
    prefix = f"Summary of other agents' responses:\n{summary}\n\nPlease re-analyze the question using these provided insights. Provide your updated answer to the question: {question}\nPresent your reasoning before final answer."
    return {"role": "user", "content": prefix}

def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def call_api_with_retry(api_call, retries=3, delay=2, jitter=1.5):
    """API调用重试机制，加入了抖动（Jitter）和动态异常处理"""
    for attempt in range(retries):
        try:
            completion = api_call()
            # 如果响应内容有效，返回结果
            if completion and completion.choices:
                return completion
            else:
                print(f"Warning: Empty response or no choices, retrying... ({attempt+1}/{retries})")
        except Exception as e:
            print(f"Error occurred: {e}, retrying... ({attempt+1}/{retries})")

        # 指数递增的等待时间 + 随机抖动
        sleep_time = delay * (2 ** attempt) + random.uniform(0, jitter)
        print(f"Waiting for {sleep_time:.2f} seconds before retrying...")
        time.sleep(sleep_time)
    
    raise Exception("Max retries exceeded. API request failed.")

if __name__ == "__main__":
    agents = 3
    rounds = 3
    random.seed(0)

    generated_description = {}

    questions = read_jsonl('./data/bioasq10b-factoid_test.jsonl')
    
    for index, data in enumerate(questions):
        question = data['question']
        context_reference = data['context']
        answer = data['answer']

        print(f"Processing question index: {index} - {question}")

        # 初始化代理上下文
        agent_contexts = [[{"role": "user", "content": """You are a skilled medical expert. Based on the following reference context: {}. what is the most appropriate answer to the following question: {} The answer should be a concise sentence or short phrase, and place it in \\boxed{{}}, at the end of your response. In addition to your answer, please provide a confidence score (0%-100%) that reflects how certain you are in the correctness of your response. Provide a logical explanation and reasoning before giving your final answer.""".format(context_reference, question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    # 组合其他代理的上下文作为信息
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                # 定义API调用
                api_call = lambda: client.chat.completions.create(
                    model="deepseek-v3",
                    messages=agent_context,
                    n=1
                )
                

                # 调用API并加入重试机制
                completion = call_api_with_retry(api_call, retries=5, delay=2, jitter=1.5)

                # 处理返回的助手消息
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    # 输出结果保存
    output_path = '../MedARC/output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    file_path = os.path.join(output_path, "factoid_ds_{}_{}.json".format(agents, rounds))

    json.dump(generated_description, open(file_path, "w"))

    print(answer)
    print(agent_context)