import os
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "Your api key"

def get_response_from_openai(prompt, model_name="gpt-4o", base_link="https://pro.aiskt.com/v1"):
    """
    获取 OpenAI 模型的响应。
    :param prompt: 输入的提示文本
    :param model_name: 使用的模型名称: ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "chatgpt-4o-latest"]
    :param base_link: OpenAI API 的基础链接，默认为 "https://pro.aiskt.com/v1"
    :return: 模型的响应内容和输入文本的长度
    """
    length = len(prompt)  # 计算输入文本的长度（可以用于计算 token 成本）

    # 检查环境变量中是否设置了 OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # 初始化 OpenAI 客户端并设置基础链接
    client = OpenAI(api_key=api_key, base_url=base_link)

    try:
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        response = None

    return response, length

# 示例用法
if __name__ == "__main__":
    prompt = "Tell me a joke about AI."
    model_name = "gpt-4o"
    base_link = "https://pro.aiskt.com/v1"  # comstomize API LINK
    response, length = get_response_from_openai(prompt, model_name, base_link)
    print(f"Response: {response}")
    print(f"Length: {length}")