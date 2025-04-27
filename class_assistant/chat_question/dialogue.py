import streamlit as st
import time
from class_assistant.chat_question.localApp import Chat_Bot

# 初始化 Chat_Bot 实例，保存在 session_state 中，防止刷新丢失
if "chat_bot" not in st.session_state:
    st.session_state.chat_bot = Chat_Bot()

# Streamed response emulator
def response_generator(user_query):
    response = st.session_state.chat_bot.chat(user_query)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def app():
    # ✅ 初始化 session_state 变量
    if "chat_bot" not in st.session_state:
        st.session_state.chat_bot = Chat_Bot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_db_ready" not in st.session_state:
        st.session_state.vector_db_ready = False
    # 设置页面标题与样式
    st.markdown("""
        <style>
        .title {
            color: #ea580c;
            font-size:50px;
        }
        .stButton>button {
            width: 100%;
            border: none;
            color: #ffedd5;
            padding: 10px;
            text-align: center;
            font-size: 20px;
            margin: 4px 2px;
            cursor: pointer;
            background-color: #f97d1c;
            border-radius: 20px;
            border: 1px solid #ccc;
        }
        .stButton>button:hover {
            background-color: #f2cac9;
            transform: scale(1.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">🤖 智能问答助教</h1>', unsafe_allow_html=True)

    # 侧边栏上传文件
    with st.sidebar:
        input_file = st.file_uploader(label='参考教材', type=['pdf'])
        if input_file is not None:
            submit_button = st.button(label='提交')
            if submit_button:
                with st.spinner("正在处理文件..."):
                    time.sleep(2)
                    st.session_state.chat_bot.createVectorDB(input_file)
                    st.session_state.vector_db_ready = True
                    st.success("文件上传成功！")

    # 初始化对话记录和状态
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_db_ready" not in st.session_state:
        st.session_state.vector_db_ready = False

    # 展示聊天历史记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入处理
    if user_query := st.chat_input("请输入..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 处理响应
        with st.chat_message("assistant"):
            if st.session_state.vector_db_ready:
                response = st.write_stream(response_generator(user_query))
            else:
                response = st.session_state.chat_bot.chat_without_file(user_query)
                st.markdown(response)

        # 添加 AI 消息
        st.session_state.messages.append({"role": "assistant", "content": response})
