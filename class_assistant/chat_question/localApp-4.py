import time
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import fitz

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

    def extract_text_from_pdf(self, file):
        '''使用 PyMuPDF 提取文本'''
        paragraphs = []
        doc = fitz.open(stream=file.read(), filetype="pdf")  # 直接从上传的 file 对象读取
        for page in doc:
            text = page.get_text("text")
            chunks = text.split('\n\n')  # 按段落分
            for chunk in chunks:
                para = chunk.strip().replace('\n', ' ')
                if len(para) > 30:
                    paragraphs.append(para)
        return paragraphs


# -------------------------------
# 滑动窗口嵌入（处理长文本）
def sliding_window_embedding(text, window_size=512, step_size=256):
    """
    对长文本进行滑动窗口切分，每个窗口生成一个嵌入向量。
    window_size：每个窗口的大小（词数）
    step_size：窗口移动步长（词数）
    """
    words = text.split()
    embeddings = []
    for start_idx in range(0, len(words), step_size):
        end_idx = min(start_idx + window_size, len(words))
        window = ' '.join(words[start_idx:end_idx])
        embedding = get_embeddings([window])[0]  # 获取嵌入向量
        embeddings.append(embedding)
    return embeddings


# -------------------------------
# 修改后的嵌入部分：使用远程接口获取文本嵌入
def get_embeddings(texts, batch_size=20):
    """
    分批调用远程嵌入接口，避免超过单次最大支持数量（如25条）。
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model="bge-m3",  # 你使用的嵌入模型
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 嵌入出错: {e}")
            all_embeddings.extend([None] * len(batch))  # 或者抛异常
    return all_embeddings


# -------------------------------
# ChromaDB 连接类
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 创建或获取 collection，并使用 HNSW 索引
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        embeddings = self.embedding_fn(documents)  # 使用嵌入函数
        self.collection.add(
            documents=documents,
            embeddings=embeddings,  # 传递文档的嵌入
            ids=[f"id{i}" for i in range(len(documents))]
        )

    def search(self, query, top_n):
        # 获取查询的嵌入向量
        query_embedding = self.embedding_fn([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
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
            "content": "你是一个善于直接简洁回答的中文助教。请直接回答用户的问题，不要输出思考过程。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model="internlm3-8b-instruct",        # 如需更换为其他模型，可选择 DeepSeek-R1-Distill-Qwen-32B、InternLM3-8B-Instruct 等
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
        self.llm_api = get_completion
        self.n_results = n_results

    def createVectorDB(self, file):
        print(file)
        pdf_loader = PDFFileLoader(file)
        # 使用滑动窗口方法来处理长文本
        paragraphs = pdf_loader.getParagraphs()
        # 创建向量数据库对象
        self.vector_db = MyVectorDBConnector("demo", get_embeddings)
        # 批量添加文档
        self.vector_db.add_documents(paragraphs)

    def chat(self, user_query):
        # 1. 检索相关文档
        search_results = self.vector_db.search(user_query, self.n_results)
        print(search_results)
        # 2. 构建 Prompt
        flattened_docs = [''.join(doc) if isinstance(doc, list) else str(doc) for doc in search_results['documents']]
        prompt = build_prompt(prompt_template, info='\n'.join(flattened_docs), query=user_query)
        # 3. 调用大模型生成回答
        response = self.llm_api(prompt)
        return response

    def chat_without_file(self, user_query):
        prompt = build_prompt(prompt_template, info="无相关教材信息，用户问题是通用性问题", query=user_query)
        response = self.llm_api(prompt)
        return response
