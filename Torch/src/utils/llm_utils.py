import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

code_model_client = OpenAI(
    api_key=os.getenv("CODE_MODEL_API_KEY", os.getenv("LLM_API_KEY", "YOUR_CODE_MODEL_API_KEY")),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = "You are a specialized code model with deep expertise in PyTorch"


def call_llm(user_prompt, examples=[], stop_str="```"):
    messages = [{"role": "system", "content": system_prompt},]

    for _in, _out in examples:
        few_shorts = [{"role": "user", "content": _in},
                      {"role": "assistant", "content": _out}, ]
        messages.extend(few_shorts)

    query_comment = [{"role": "user", "content": user_prompt},
                     {"role": "assistant", "content": "[", "prefix": True}]
    messages.extend(query_comment)

    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=messages,
                stop=[stop_str],
                temperature=0.8,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling llm: {e}")
            time.sleep(3)
    return ''


def call_FIM_llm(user_prompt, suffix):
    response = client.completions.create(
        model="deepseek-chat",
        prompt=system_prompt + "\n" + user_prompt,
        suffix=suffix,
        max_tokens=64,
        temperature=0.8,
    )
    return response.choices[0].text


def call_code_model(user_prompt, examples=[]):
    messages = [{"role": "system", "content": system_prompt},]

    for _in, _out in examples:
        few_shorts = [{"role": "user", "content": _in},
                      {"role": "assistant", "content": _out}, ]
        messages.extend(few_shorts)

    query_comment = [{"role": "user", "content": user_prompt}]
    messages.extend(query_comment)

    try:
        response = code_model_client.chat.completions.create(
            model="qwen3-coder-30b-a3b-instruct",
            messages=messages,
            temperature=0.5,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling code model: {e}")
        print("Falling back to summary model.")
        return call_llm(user_prompt, examples)
