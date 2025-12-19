from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    input_system_prompt,
    classify_intent,
    handle_question,
    handle_chat,
)



# 时间检测器
import time
def timed_node(name, func):
    def wrapper(state):
        start = time.time()
        result = func(state)
        print(f"[{name}] 耗时: {time.time() - start:.3f}s")
        return result
    return wrapper

def build_graph():
    # 1. 创建一个基于 GraphState 的状态图
    graph = StateGraph(GraphState)

    # 2. 注册所有节点
    # graph.add_node("classify_intent", classify_intent)
    # graph.add_node("handle_question", handle_question)
    # graph.add_node("handle_chat", handle_chat)
    graph.add_node("input_system_prompt", timed_node("input_system_prompt", input_system_prompt))
    graph.add_node("classify_intent", timed_node("classify_intent", classify_intent))
    graph.add_node("handle_question", timed_node("handle_question", handle_question))
    graph.add_node("handle_chat", timed_node("handle_chat", handle_chat))

    # 3. 设置入口节点
    graph.set_entry_point("input_system_prompt")

    # 3.1 input_system_prompt -> classify_intent
    graph.add_edge("input_system_prompt", "classify_intent")

    # 4. 根据 intent 的值决定走哪个分支
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state["intent"],
        {
            "question": "handle_question",
            "chat": "handle_chat",
        }
    )

    # 5. 设置结束节点
    graph.add_edge("handle_question", END)
    graph.add_edge("handle_chat", END)

    # 6. 编译 graph
    return graph.compile()
