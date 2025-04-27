from transformers import AutoTokenizer  # 此处仅用于提示原有代码结构，嵌入调用已改为远程接口
from bigdl.llm.transformers import AutoModelForCausalLM  # 不再用于加载本地模型
import torch
import time
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# -------------------------------
# 全局初始化：用于调用文本生成与嵌入接口
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="COEZ2XUAPTLNQQOL1LUMPARXYDDCXYUYJKOHAB83",
    default_headers={"X-Failover-Enabled": "true"}
)

start_time = time.time()  # 记录开始时间

# -------------------------------
# PDF 文件加载类
class PDFFileLoader():
    def __init__(self, file) -> None:
        self.paragraphs = self.extract_text_from_pdf(file)

    def getParagraphs(self):
        return self.paragraphs

    def extract_text_from_pdf(self, filename, page_numbers=None):
        '''从 PDF 文件中（按指定页码）提取文字'''
        paragraphs = []
        buffer = ''
        full_text = ''
        # 提取全部文本
        for i, page_layout in enumerate(extract_pages(filename)):
            if page_numbers is not None and i not in page_numbers:
                continue
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text() + '\n'

        # 段落分割
        lines = full_text.split('。\n')
        for text in lines:
            buffer = text.strip(' ').replace('\n', ' ').replace('[', '').replace(']', '')
            if len(buffer) < 10:
                continue
            if buffer:
                paragraphs.append(buffer)
                buffer = ''
        if buffer and len(buffer) > 10:
            paragraphs.append(buffer)
        return paragraphs

# -------------------------------
# 修改后的嵌入部分：使用远程接口获取文本嵌入
def get_embeddings(texts):
    """
    调用昇腾资源包中的远程嵌入接口，获取文本的嵌入向量。
    texts 为一个包含一个或多个字符串的列表。
    使用模型 "bce-embedding-base_v1"，返回每个文本对应的嵌入向量列表。
    """
    try:
        response = client.embeddings.create(
            model="bce-embedding-base_v1",  # 可替换为其他资源包内嵌入模型名称
            input=texts
        )
        # 假设返回格式为 {"data": [{"embedding": [ ... ], "index": 0}, ...]}
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings
    except Exception as e:
        print(f"调用远程嵌入接口出错: {e}")
        return None

# -------------------------------
# ChromaDB 连接类
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 创建或获取 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        self.collection.add(
            documents=documents,
            ids=[f"id{i}" for i in range(len(documents))]
        )

    def search(self, query, top_n):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        return results

    def q_search(self, query, top_n):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        return results

# -------------------------------
# Prompt 拼接函数
def build_prompt(prompt_template, **kwargs):
    '''将 Prompt 模板赋值'''
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt

# -------------------------------
# Prompt 模板定义
prompt_template = """
你是一个优秀的中文智能助教。
你的任务是根据下述给定的已知信息、结合自己的知识用中文总结出问题的答案，准确地回答用户问题。
确保你的回复完全依据下述已知信息，回答的内容务必符合中文语法规范，通顺流畅，简短精炼并且准确无误。
如果下述已知信息不足以回答用户的问题，请直接回复"与教材无关，我无法回答您的问题"。

已知信息:
__INFO__

用户问：
__QUERY__

请用中文回答用户问题。
"""

prompt_question = """
你是一个优秀的中文出题助手,你的任务是根据下述给定的已知信息为用户出给定题目数量的题.
比如题目类型是选择题，题目数量是3，那么你就帮用户出3道选择题，并且每道题的都要给出正确答案，以此类推。
确保出的题目必须与用户要求、题目类型、题目数量的相符。不要胡编乱造，不要有无关的符号。
如果题目类型是判断题，那么你要出需要判断对错的题，并给出正确答案，不要说多余的话。
如果题目类型是问答题，那么你需要用中文出题，并且用中文给出正确的回答，尽量简短精炼。
如果下述已知信息不足以出题，请直接回复"与教材无关，我无法完成您的任务"。
如果能回答，那么prompt里的话请不要输出

已知信息:
__INFO__

用户要求：
__QUERY__

题目类型：
__TYPE__

题目数量：
__NUM__

请用中文表达。
"""

prompt_code = """
你是一个优秀的代码智能助手。
你的任务是扮演下述给定的角色回答用户与代码有关的问题。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

角色:
__ROLE__

用户问：
__QUERY__

请用中文回答用户问题，并且不要输出prompt里的内容。
"""

# -------------------------------
# 修改后的 get_completion 函数，调用昇腾资源包内的文本生成接口
def get_completion(prompt):
    """调用昇腾平台资源包内的 QwQ-32B 模型接口生成文本响应。"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and harmless assistant. You should think step-by-step."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model="QwQ-32B",        # 如需更换为其他模型，可选择 DeepSeek-R1-Distill-Qwen-32B、InternLM3-8B-Instruct 等
        stream=False,          # 非流式调用，返回完整响应
        max_tokens=1024,
        temperature=1,
        top_p=0.2,
        extra_body={"top_k": 50},
        frequency_penalty=0,
        messages=messages,
    )
    # 假设接口返回格式为：{"choices": [{"message": {"content": "生成的回答内容"}}]}
    return response.choices[0].message.content

# -------------------------------
# Chat_Bot 类，保持原有结构，仅 llm_api 及 get_embeddings 调用方式改变
class Chat_Bot:
    def __init__(self, n_results=2):
        # llm_api 指向新的 get_completion 函数（调用昇腾模型接口）
        self.llm_api = get_completion
        self.n_results = n_results

    def createVectorDB(self, file):
        print(file)
        pdf_loader = PDFFileLoader(file)
        # 创建向量数据库对象，使用远程接口的 get_embeddings 获取文档嵌入
        self.vector_db = MyVectorDBConnector("demo", get_embeddings)
        self.vector_db.add_documents(pdf_loader.getParagraphs())

    def chat(self, user_query):
        # 1. 检索相关文档
        search_results = self.vector_db.search(user_query, self.n_results)
        print(search_results)
        # 2. 构建 Prompt
        prompt = build_prompt(prompt_template, info=search_results['documents'][0], query=user_query)
        print("prompt===================>")
        print(prompt)
        # 3. 调用大模型生成回答
        response = self.llm_api(prompt)
        return response

    def question(self, user_query, type, num):
        search_results = self.vector_db.q_search(user_query, 5)
        prompt = build_prompt(prompt_question, info=search_results['documents'][0],
                              query=user_query, type=type, num=num)
        print("prompt===================>")
        print(prompt)
        response = self.llm_api(prompt)
        return response

    def code(self, role, user_query):
        prompt = build_prompt(prompt_code, role=role, query=user_query)
        print("prompt===================>")
        print(prompt)
        response = self.llm_api(prompt)
        return response

# -------------------------------
# 注意：已移除本地模型加载代码，不再调用 AutoModelForCausalLM.load_low_bit 和 AutoTokenizer.from_pretrained，
# 嵌入向量和大模型调用均使用远程接口

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time
print(f"执行时间为: {elapsed_time:.4f} 秒")
