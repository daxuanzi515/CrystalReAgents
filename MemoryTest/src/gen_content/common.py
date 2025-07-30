import os
import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

os.environ["CREWAI_STORAGE_DIR"] = "./crew_memory"


load_dotenv(override=True)

# 初始化 LLM
qwen = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("Temperature", 0.3)),
    max_tokens=int(os.getenv("max_tokens", 2048)),
    streaming=False,
    timeout=60
)

def generate_log_fileName():
    """
    根据当前时间生成日志文件路径
    Returns:
        str: 完整的日志文件路径
    """
    # 日志目录，根据自己项目修改
    log_dir = "src/gen_content/log"  
    os.makedirs(log_dir, exist_ok=True)
    # 生成精确到秒的时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # 返回完整日志文件路径
    return os.path.join(log_dir, f'crewai_{timestamp}.log')

from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from memo_monitor import *
def setup_memory_event_monitors():
    for listener in [
        MemoryLogger(),
        MemoryPerformanceMonitor(),
        MemoryErrorTracker(notify_email=None)
    ]:
        listener.setup_listeners(crewai_event_bus)


# 定义五个内容生成相关 Agent
decomposer = Agent(
    role="内容主题分解专家",
    goal="从用户输入中提取多个清晰子话题",
    backstory="你是一位擅长内容架构设计的专家，能精准分解复杂请求。",
    llm=qwen,
    memory=True,
)

researcher = Agent(
    role="网页信息收集者",
    goal="为每个子话题收集权威信息摘要",
    backstory="你负责为每个子话题找到权威信息来源并存入记忆。",
    llm=qwen,
    memory=True
)

writer = Agent(
    role="主题段落撰写者",
    goal="根据记忆中的资料撰写内容段落",
    backstory="你是一位经验丰富的内容编辑，擅长从资料中提炼观点。",
    llm=qwen,
    memory=True
)

refiner = Agent(
    role="风格逻辑校正员",
    goal="统一各段落语言风格和逻辑顺序",
    backstory="你是语言风格与表达清晰度的把关人。",
    llm=qwen,
    memory=True
)

assembler = Agent(
    role="最终报告输出者",
    goal="组合所有段落并输出为纯文本格式",
    backstory="你是文档编辑器，专注格式规范与内容完整性。",
    llm=qwen,
    memory=True
)

task1 = Task(
    description="你需要分析用户输入的调研请求内容：{input}，将其拆解为若干子话题。如果发现某些话题边界模糊或语义重复，请与Agent2（信息收集者）进行讨论并投票确认最终子目标列表。",
    expected_output="子话题列表，格式为 JSON",
    agent=decomposer
)

task2 = Task(
    description="针对记忆中的每个子话题爬取网页资料摘要，并按话题写入记忆。",
    expected_output="每个话题对应的网页摘要",
    agent=researcher
)

task3 = Task(
    description="根据话题和摘要内容撰写完整段落。",
    expected_output="每个话题一个段落",
    agent=writer
)

task4 = Task(
    description="统一所有段落的逻辑结构和语言风格，生成最终版本。",
    expected_output="6个统一风格的段落",
    agent=refiner
)

task5 = Task(
    description="组合所有段落并输出纯文本格式的报告，不使用 Markdown。",
    expected_output="完整的纯文本报告",
    agent=assembler
)


if __name__ == "__main__":
    ## monitor
    setup_memory_event_monitors()

    # 创建 Crew
    crew = Crew(
        agents=[decomposer, researcher, writer, refiner, assembler],
        tasks=[task1, task2, task3, task4, task5],
        process=Process.sequential,
        verbose=True,
        output_log_file=generate_log_fileName() 
    )

    # 启动流程
    result = crew.kickoff(inputs={
        "input": """根据如下信息生成一份关于"郁金香泡沫"的调查报告用于制作Slides，其中要覆盖到以下内容：
    labubu热潮、爆裂熊的退潮、高定奢侈品、明星炒作、黄牛乱象、讨论与思考。
    最后返回给用户纯文本内容1000字左右的报告，不能写成Markdown格式。"""
    })

    # 打印结果
    print("\n🎯 最终输出内容：\n")
    print(result)
