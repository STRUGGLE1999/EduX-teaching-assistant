from openai import OpenAI
import time

# ---------------------------
# 初始化远程接口客户端（使用昇腾资源包接口）
client = OpenAI(
    base_url="https://ai.gitee.com/v1",  # 昇腾资源包的 API 网关地址
    api_key="COEZ2XUAPTLNQQOL1LUMPARXYDDCXYUYJKOHAB83",  # 替换为你的访问密钥
    default_headers={"X-Failover-Enabled": "true"}
)

start_time = time.time()  # 记录初始化开始时间

def generate_text_from_model(prompt, max_tokens=4096):
    """
    通过远程接口生成文本。调用格式模仿 OpenAI 的 ChatCompletion 接口，
    使用大模型 "QwQ-32B"（或其他你在昇腾资源包中选择的模型），
    生成最大 1024 个新 tokens。
    """
    # 构造对话格式消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # 调用远程接口生成文本（非流式调用）
    response = client.chat.completions.create(
        model="internlm3-8b-instruct",     # 可根据实际情况调整模型名称，例如 DeepSeek-R1-Distill-Qwen-32B
        stream=False,
        max_tokens=1024,
        temperature=1,
        top_p=0.2,
        extra_body={"top_k": 50},
        frequency_penalty=0,
        messages=messages
    )
    
    # 修改返回结果的访问方式，由 response["choices"][0]["message"]["content"]
    # 改为属性访问：response.choices[0].message.content
    return response.choices[0].message.content

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time
print(f"初始化及接口配置耗时: {elapsed_time:.4f} 秒")

# # ------------- 测试文本生成 -------------
# test_prompt = "介绍一下人工智能的发展历程。"
# generated_text = generate_text_from_model(test_prompt)
# print("生成的文本：", generated_text)
