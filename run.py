import subprocess
import os
import socket
import sys

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    # base_dir = os.path.abspath(".")

    ports = [6688, 8866]
    for port in ports:
        if is_port_in_use(port):
            print(f"❌ 端口 {port} 已被占用，请先释放或修改 Streamlit 应用端口。")
            return

    # 启动第一个 Streamlit 应用（问答/出题）
    # chat_dir = os.path.join(base_dir, "class_assistant", "chat_question")
    subprocess.Popen(
        f"streamlit run class_assistant/chat_question/OpenEduECNU1.py --server.port 6688",
        shell=True
    )

    # 启动第二个 Streamlit 应用（大纲）
    # courseware_dir = os.path.join(base_dir, "class_assistant", "courseware")
    subprocess.Popen(
        f"streamlit run class_assistant/courseware/OpenEduECNU2.py --server.port 8866",
        shell=True
    )

    input("按回车键退出主程序，但 Streamlit 应用仍在后台运行...")

if __name__ == "__main__":
    main()
