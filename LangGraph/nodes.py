from state import GraphState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# =========================
# 1. 加载 .env 中的环境变量
# =========================
load_dotenv()

# =========================
# 2. 初始化 DeepSeek LLM
# =========================
llm = ChatOpenAI(
    model="deepseek-reasoner",   # DeepSeek 深度思考模型
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0                # 分类任务必须稳定
)

# =========================
# 节点 1：用户输入大纲提示词
# =========================
def input_system_prompt(state: GraphState) -> GraphState:
    """让用户输入自定义提示词"""
    prompt = input("请输入自定义提示词: ").strip()
    state["user_input"] = prompt if prompt else None
    return state


# =========================
# 节点 2：意图判断（LLM 版）
# =========================
def classify_intent(state: GraphState) -> GraphState:
    """
    使用 LLM 判断用户意图
    只能返回：question 或 chat
    """

    user_input = state["user_input"]

    prompt = f"""
你是一个【意图分类器】，不是聊天机器人。

任务：
- 判断用户输入是【question】还是【chat】

规则：
- 如果用户在询问信息、方法、原因 → question
- 如果只是寒暄、情绪表达 → chat
- 你【只能】返回一个单词：question 或 chat
- 不要解释，不要多余内容

用户输入：
{user_input}
"""

    try:
        result = llm.invoke(prompt).content.strip().lower()

        # ---- 兜底保护（非常重要）----
        if result not in ("question", "chat"):
            result = "chat"

        state["intent"] = result

    except Exception as e:
        # LLM 出问题时，流程仍然能跑
        print("LLM 调用失败：", e)
        state["intent"] = "chat"

    return state


# =========================
# 节点 3：处理问题
# =========================
def handle_question(state: GraphState) -> GraphState:
    state["response"] = "我判断这是一个【问题】，后续可以进入正式问答流程。"
    return state


# =========================
# 节点 4：处理聊天
# =========================
def handle_chat(state: GraphState) -> GraphState:
    state["response"] = "我判断这是【聊天】，可以轻松一点回复。"
    return state
