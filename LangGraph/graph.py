from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import GraphState
from nodes import (
    input_system_prompt,
    concat_prompt,
    generate_outline,
    outline_human_intervention
)
import os

# 状态存储路径
STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "state")
os.makedirs(STATE_DIR, exist_ok=True) # 确保状态目录存在
CHECKPOINT_DB = os.path.join(STATE_DIR, "checkpoints.db") # 状态检查点数据库路径


# 时间检测器
import time
def timed_node(name, func):
    def wrapper(state):
        start = time.time()
        result = func(state)
        print(f"[{name}] 耗时: {time.time() - start:.3f}s")
        return result
    return wrapper


# 构建状态图
def build_graph():
    # 1. 创建一个基于 GraphState 的状态图
    graph = StateGraph(GraphState)

    # 2. 注册所有节点
    graph.add_node("input_system_prompt", timed_node("input_system_prompt", input_system_prompt))
    graph.add_node("concat_prompt", timed_node("concat_prompt", concat_prompt))
    graph.add_node("generate_outline", timed_node("generate_outline", generate_outline))
    graph.add_node("outline_human_intervention", timed_node("outline_human_intervention", outline_human_intervention))

    # 3. 设置入口节点
    graph.set_entry_point("input_system_prompt")

    # 4. 设置边：节点1 -> 节点2 -> 节点3
    graph.add_edge("input_system_prompt", "concat_prompt")
    graph.add_edge("concat_prompt", "generate_outline")

    # 5. 大纲生成后进入节点4
    graph.add_edge("generate_outline", "outline_human_intervention")
    
    # 6. 人工干预后条件边：quit->END，y->节点1，其他->节点2
    graph.add_conditional_edges(
        "outline_human_intervention",
        lambda state: END if state["user_input"].lower() == "quit" else ("input_system_prompt" if state["user_input"].lower() == "y" else "concat_prompt")
    )

    # 7. 返回 graph (不带 checkpointer，在 main.py 中使用上下文管理器)
    return graph

def get_checkpointer():
    """获取 checkpointer 上下文管理器"""
    return SqliteSaver.from_conn_string(CHECKPOINT_DB)
