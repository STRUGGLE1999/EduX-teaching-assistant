import os
import streamlit as st
from streamlit_option_menu import option_menu

# 导入各个模块
from class_assistant.chat_question import dialogue, question
from class_assistant.courseware import outlineGenerate, pptGenerate

import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# 设置页面基本配置
st.set_page_config(
    page_title="EduX 多功能智课伴侣",
    page_icon=os.path.join("class_assistant", "pic", "EduX.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义侧边栏样式（可选）
st.markdown("""
    <style>
        .sidebar.sidebar-content {
            background-color: #add8e6;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image(os.path.join("class_assistant", "pic", "EduX_logo_1.png"), use_container_width=True)
    
    st.caption("<p align='right'>designed by EduX</p>", unsafe_allow_html=True)
    
    # 定义顶级导航菜单
    selected_page = option_menu(
        "导航菜单",
        options=["问答与出题", "智慧教学"],
        icons=["chat", "book"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#043d6b"}
        }
    )

# 根据顶级导航切换不同模块
if selected_page == "问答与出题":
    st.title("智能问答与出题")
    # 进一步设置子菜单：比如“智能问答”、“智能出题”
    sub_option = st.radio("选择功能", ["智能问答", "智能出题"])
    if sub_option == "智能问答":
        dialogue.app()   # 调用 dialogue 模块中的 app() 函数
    elif sub_option == "智能出题":
        question.app()   # 调用 question 模块中的 app() 函数
elif selected_page == "智慧教学":
    st.title("智慧教学")
    # 在智慧教学模块中，可以再细分为“智慧教学大纲”和“智慧教学 PPT”
    sub_option = st.radio("选择功能", ["智慧教学大纲", "智慧教学ppt"])
    if sub_option == "智慧教学大纲":
        outlineGenerate.app()   # 调用 outlineGenerate 模块的 app() 函数
    elif sub_option == "智慧教学ppt":
        pptGenerate.app()       # 调用 pptGenerate 模块的 app() 函数
