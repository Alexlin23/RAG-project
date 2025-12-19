from state import GraphState

# =========================
# 节点 1：意图判断
# =========================
def classify_intent(state: GraphState) -> GraphState:
    """
    这个节点只负责一件事：
    根据用户输入，判断是 question 还是 chat
    （当前版本：不用 AI，用 if/else）
    """
    text = state["user_input"]

    # 非常粗糙的规则，但足够用于学习
    if "?" in text or "怎么" in text or "如何" in text:
        state["intent"] = "question"
    else:
        state["intent"] = "chat"

    return state


# =========================
# 节点 2：处理问题
# =========================
def handle_question(state: GraphState) -> GraphState:
    """
    这个节点只处理 question 分支
    """
    state["response"] = "我判断这是一个【问题】，后续可以进入正式问答流程。"
    return state


# =========================
# 节点 3：处理聊天
# =========================
def handle_chat(state: GraphState) -> GraphState:
    """
    这个节点只处理 chat 分支
    """
    state["response"] = "我判断这是【聊天】，可以轻松一点回复。"
    return state
