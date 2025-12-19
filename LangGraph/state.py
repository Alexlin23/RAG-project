from typing import TypedDict, Optional

class GraphState(TypedDict):
    """
    整个 LangGraph 流程中共享的状态数据
    所有 node 只能通过它来读写信息
    """
    user_input: str            # 用户原始输入
    intent: Optional[str]      # 判断出的意图：question / chat
    response: Optional[str]    # 最终要返回给用户的内容
