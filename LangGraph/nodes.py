from state import GraphState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sripts.tool import read_all_texts_in_dir
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
    temperature=0.8,                # 文本生成温度
    max_tokens=64000
)

# =========================
# 节点 1：用户输入提示词
# =========================
def input_system_prompt(state: GraphState) -> GraphState:
    """让用户输入自定义提示词"""
    prompt = input("请输入自定义提示词: ").strip()
    state["user_input"] = prompt if prompt else None
    return state


# =========================
# 节点 2：拼接提示词（大纲）
# =========================
# 系统提示词（大纲生成指令）
OUTLINE_SYSTEM_PROMPT = """你是一位专业的网文小说作家，擅长撰写超长篇小说大纲。

【大纲要求】
1. 每次生成15章内容的大纲
2. 大纲篇幅至少为这批章节总正文的30%（约 13500 字）
3. 不需要解释，不包含章节标题，不要按章节分段
4. 连贯地写下来，保持情节流畅
5. 承接已有大纲的剧情，保证前后伏笔一致
6. 可拓展世界观，但不能与已有设定冲突

请直接输出大纲内容，不要包含任何额外的说明或标题。"""

def concat_prompt(state: GraphState) -> GraphState:
    """拼接提示词，使用 DeepSeek 消息格式"""
    if state["first_time"] is True:
        # 首次调用：读取 data/outline 下所有文件内容
        outline_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outline")
        outline_contents = read_all_texts_in_dir(outline_dir)
        outline_text = "\n\n".join(outline_contents)
        # 结合大纲和用户输入生成首次提示词（DeepSeek 格式）
        state["prompts_message"] = [
            {"role": "system", "content": OUTLINE_SYSTEM_PROMPT},
            {"role": "user", "content": f"过去章节大纲：\n{outline_text}"},
            {"role": "user", "content": state["user_input"]}
        ]
    else:
        # 非首次：将 user_input 追加到 messages 列表
        state["prompts_message"].append({"role": "user", "content": state["user_input"]})
    return state


# =========================
# 节点 3：大纲生成（LLM）
# =========================
def generate_outline(state: GraphState) -> GraphState:
    """调用 LLM 生成大纲"""
    response = llm.invoke(state["prompts_message"])
    state["response"] = response.content
    print(state["response"])
    state["first_time"] = False

    print("大纲已生成")
    return state


# =========================
# 节点 4：大纲人工干预
# =========================
def outline_human_intervention(state: GraphState) -> GraphState:
    """人工干预节点：用户输入y重新生成，输入其他文本作为反馈追加到消息列表"""
    state["user_input"] = input("请输入指令(输入'y'重新生成，或输入其他文本作为反馈): ").strip()
    
    if state["user_input"].lower() == "y":
        state["first_time"] = True
    else:
        # 将response作为ai回复，用户输入作为新的user_input追加到prompts_message
        state["prompts_message"].append({"role": "assistant", "content": state["response"]})
    
    return state