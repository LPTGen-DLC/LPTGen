import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

system_prompt = (
    "You are a specialized code Large Language Model (LLM) with deep expertise in Apache TVM. "
    "Your knowledge spans topi, te, tir, scheduling primitives"
)


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
                model="deepseek-chat",
                messages=messages,
                stop=[stop_str],
                temperature=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] failed in call LLM due to: {e}")
            time.sleep(5)
    return ''
