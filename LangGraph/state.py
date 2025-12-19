from typing import TypedDict, Optional, List, Dict

class GraphState(TypedDict):
    """
    整个 LangGraph 流程中共享的状态数据
    所有 node 只能通过它来读写信息
    """
    user_input: str            # 用户输入
    first_time: bool           # 是否第一次调用
    prompts_message: Optional[List[Dict[str, str]]]    # 消息列表
    response: Optional[str]        # LLM 的响应
    chapter_progress: Optional[int] # 章节进度
