from crewai import Agent, Task, Crew
from crewai.memory import Mem0Memory

# Step 1: 创建一个 Mem0Memory 实例（长期记忆插件）
memory = Mem0Memory(
    save_path="memories/agent1_memory.json",   # 本地存储路径
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # 你也可以用 openai
)

# Step 2: 创建 Agent，并绑定长期记忆
agent = Agent(
    name="Researcher",
    role="AI scientist",
    goal="深入理解用户的提问内容并持续学习",
    memory=memory,         # ⚠️绑定长期记忆
    allow_delegation=False,
    verbose=True
)

# Step 3: 创建任务
task = Task(
    description="请解释什么是强化学习",
    agent=agent
)

# Step 4: 创建 Crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

# Step 5: 执行任务
result = crew.run()
print("任务结果：", result)
