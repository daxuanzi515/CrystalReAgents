# my demo scripts
import openai
# Qwen-2.5-72b-instruct
def get_chat_completions(messages, api_key, base_url, use_llm_model):
    """
    使用 openai 接口调用大模型。
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(model=use_llm_model, messages=messages)

    return response

def check_money():
  import requests
  url = "https://wcode.net/api/account/billing/grants"
  payload = {}
  headers = {
    'Authorization': 'Bearer sk-581.ti7EqhYyTtdWTA5dXk0th5Yootg9iKFTVayIOEIXVk3xuODP'  # <-------- TODO: 替换这里的 API_KEY
  }
  response = requests.request("GET", url, headers=headers, data=payload)
  print(response.text)
  return response.text

# print(openai.proxy)
# usage example
messages = [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "你好"
    }
  ]

api_key = "sk-581.ti7EqhYyTtdWTA5dXk0th5Yootg9iKFTVayIOEIXVk3xuODP"
base_url = "https://wcode.net/api/gpt/v1"
use_llm_model = "qwen2.5-72b-instruct"

# run something
# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8800'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8800'

# response = get_chat_completions(messages, api_key, base_url, use_llm_model)
# reply_content = response.choices[0].message.content
# print(reply_content)

# money ?
check_money()