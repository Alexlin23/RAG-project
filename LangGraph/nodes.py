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
# 3. 初始化参数
# =========================
outline_length = 10 # 大纲长度，每次生成10章内容的大纲
chapter_length = 3000 # 章节长度，每次生成3000字的内容


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
OUTLINE_SYSTEM_PROMPT = f"""你是一位专业的网文小说作家，擅长撰写超长篇小说大纲。

【大纲要求】
1. 每次生成{outline_length}章内容的大纲
2. 大纲篇幅至少为{outline_length * chapter_length * 0.3}字
3. 不需要解释，不包含章节标题，不要按章节分段
4. 连贯地写下来，保持情节流畅
5. 承接已有大纲的剧情，保证前后伏笔一致
6. 可拓展世界观，但不能与已有设定冲突
7. 情节不要写太细致，只写大体框架
8. 写清楚地点，设定，技能，角色名称以及那些特殊设定名词
9. 检查字数一定要符合标准

请直接输出大纲内容，不要包含任何额外的说明或标题。"""

def concat_prompt(state: GraphState) -> GraphState:
    """拼接提示词，使用 DeepSeek 消息格式"""
    if state["first_time"] is True:
        # 首次调用：读取 data/outline 下所有文件内容
        outline_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outline")
        outline_contents = read_all_texts_in_dir(outline_dir)
        outline_text = "\n\n".join(outline_contents)
        
        # 读取全部设定集
        worldguide_dir = os.path.join(os.path.dirname(__file__), "..", "data", "WorldGuide")
        if os.path.isdir(worldguide_dir):
            worldguide_contents = read_all_texts_in_dir(worldguide_dir)
            worldguide_text = "\n\n".join(worldguide_contents)
        else:
            worldguide_text = ""
        
        # 结合大纲、设定集和用户输入生成首次提示词（DeepSeek 格式）
        state["prompts_message"] = [
            {"role": "system", "content": OUTLINE_SYSTEM_PROMPT},
            {"role": "user", "content": f"过去章节的大纲：\n{outline_text}"},
            {"role": "user", "content": f"全部设定集：\n{worldguide_text}"},
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
    
    state["first_time"] = False
    
    outline_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outline")
    with open(os.path.join(outline_dir,
     f"outline_{state['chapter_progress']}-{state['chapter_progress']+outline_length-1}.txt"),
     "w", encoding="utf-8") as f:
        f.write(state["response"])

    print("大纲已生成")
    return state


# =========================
# 节点 4：大纲人工干预
# =========================
def outline_human_intervention(state: GraphState) -> GraphState:
    """人工干预节点：用户输入y重新生成，输入其他文本作为反馈追加到消息列表"""
    state["user_input"] = input("请输入指令(输入'y'重新生成, 'quit'退出, 或输入其他文本作为反馈): ").strip()
    
    if state["user_input"].lower() == "y":
        state["first_time"] = True
        state["chapter_progress"] += outline_length # 增加章节进度
    else:
        # 将response作为ai回复，用户输入作为新的user_input追加到prompts_message
        state["prompts_message"].append({"role": "assistant", "content": state["response"]})
    
    return state


# =========================
# 节点 5：设定集提示词拼接
# =========================
# 系统提示词（设定集生成指令）
WORLDGUIDE_SYSTEM_PROMPT = """你是一位专业的网文小说作家，擅长构建完整的世界观和角色设定。

【设定集要求】
1. 根据大纲内容提取并补充世界观、角色、势力、道具等设定
2. 设定要详细且符合剧情发展需要
3. 新设定不能与已有设定冲突，如有冲突优先改动新设定
4. 设定可拓展，但要保持一致性
5. 设定格式清晰，便于后续参考

请直接输出设定集内容，不要包含任何额外的说明。"""


def concat_worldguide_prompt(state: GraphState) -> GraphState:
    """拼接设定集生成提示词，使用 DeepSeek 消息格式"""
    
    # 1. 读取最新大纲（当前批次：chapter_progress 到 chapter_progress+outline_length-1）
    outline_dir = os.path.join(os.path.dirname(__file__), "..", "data", "outline")
    latest_outline_file = os.path.join(
        outline_dir,
        f"outline_{state['chapter_progress']-outline_length}-{state['chapter_progress']-1}.txt"
    ) # 最新大纲文件路径
    if os.path.exists(latest_outline_file):
        with open(latest_outline_file, "r", encoding="utf-8") as f: # 检查文件是否存在
            latest_outline = f.read() # 读取最新大纲文件内容
    else:
        latest_outline = ""
    
    # 2. 读取过往设定集
    worldguide_dir = os.path.join(os.path.dirname(__file__), "..", "data", "WorldGuide")
    if os.path.isdir(worldguide_dir): # 检查文件夹是否存在
        worldguide_contents = read_all_texts_in_dir(worldguide_dir) # 读取所有文本文件内容
        past_worldguide = "\n\n".join(worldguide_contents) # 将所有设定集内容用双换行符连接
    else:
        past_worldguide = ""
    
    # 3. 构建消息列表（system放指令，user放动态内容）
    if state["first_time"] is True:
        # 首次调用：构建完整消息列表
        state["prompts_message"] = [
            {"role": "system", "content": WORLDGUIDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"【过往设定集】\n{past_worldguide}"},
            {"role": "user", "content": f"【最新大纲】\n{latest_outline}"},
            {"role": "user", "content": state["user_input"]}
        ]
    else:
        # 非首次：将 user_input 追加到 messages 列表
        state["prompts_message"].append({"role": "user", "content": state["user_input"]})
    
    return state


# =========================
# 节点 6：设定集生成（LLM）
# =========================
def generate_worldguide(state: GraphState) -> GraphState:
    """调用 LLM 生成设定集"""
    response = llm.invoke(state["prompts_message"])
    state["response"] = response.content
    
    state["first_time"] = False
    
    # 保存设定集文件
    worldguide_dir = os.path.join(os.path.dirname(__file__), "..", "data", "WorldGuide")
    os.makedirs(worldguide_dir, exist_ok=True)  # 确保目录存在
    
    worldguide_file = os.path.join(
        worldguide_dir,
        f"worldguide_{state['chapter_progress']}-{state['chapter_progress']+outline_length-1}.txt"
    )
    with open(worldguide_file, "w", encoding="utf-8") as f:
        f.write(state["response"])
    
    print("设定集已生成")
    return state


# =========================
# 节点 7：设定集人工干预
# =========================
def worldguide_human_intervention(state: GraphState) -> GraphState:
    """人工干预节点：用户输入y进入下一流程，输入其他文本作为反馈追加到消息列表"""
    state["user_input"] = input("请输入指令(输入'y'确认进入下一流程, 'quit'退出, 或输入其他文本作为反馈): ").strip()
    
    if state["user_input"].lower() == "y":
        state["first_time"] = True
    else:
        # 将response作为ai回复，用户输入作为新的user_input追加到prompts_message
        state["prompts_message"].append({"role": "assistant", "content": state["response"]})
    
    return state