import time
import random
import json
import os
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url=""
)

def construct_message(question, context_reference):
    return {
        "role": "user",
        "content": """You are a skilled medical expert. Based on the following reference context: {}. what is the most appropriate answer to the following question: {} The answer should consist of one to two complete sentences, and place it in \\boxed{{}}, at the end of your response.""".format(context_reference, question)
    }

def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def call_api_with_retry(api_call, retries=3, delay=2, jitter=1.5):
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
    agents = 1
    rounds = 1
    random.seed(0)

    generated_description = {}

    questions = read_jsonl('./data/bioasq12b-summary_test.jsonl')
    
    for index, data in enumerate(questions):
        question = data['question']
        context_reference = data['context']
        answer = data['answer']

        print(f"Processing question index: {index} - {question}")

        # 只生成一次消息，而不涉及多轮讨论或其他代理的回答
        agent_context = [construct_message(question, context_reference)]

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

        generated_description[question] = (agent_context, answer)

    # 输出结果保存
    output_path = '../MedARC/output'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    file_path = os.path.join(output_path, "open-ended_0ds_{}_{}.json".format(agents, rounds))

    json.dump(generated_description, open(file_path, "w"))

    print(answer)
    print(agent_context)