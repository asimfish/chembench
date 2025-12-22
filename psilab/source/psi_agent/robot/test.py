import os

def list_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print(f"目录: {item}")
        else:
            print(f"文件: {item}")

# 使用示例
directory_path = os.path.dirname(os.path.abspath(__file__))  # 替换为你的目录路径
list_directory(directory_path)