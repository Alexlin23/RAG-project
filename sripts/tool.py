# tool.py
# 最基础的工具函数集合

import os
from typing import List


def read_text_file(file_path: str) -> str:
    """
    读取单个文本文件并返回内容

    参数：
    - file_path: 文件的完整路径

    返回：
    - 文件内容字符串
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_all_texts_in_dir(dir_path: str) -> List[str]:
    """
    读取目录下所有 .txt 文件的内容（不递归）

    参数：
    - dir_path: 目录路径

    返回：
    - 按文件名排序后的文本内容列表
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"不是有效目录: {dir_path}")

    contents: List[str] = []

    for filename in sorted(os.listdir(dir_path)):
        if filename.lower().endswith(".txt"):
            full_path = os.path.join(dir_path, filename)
            contents.append(read_text_file(full_path))

    return contents
