# # 下载模型文件
# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model", local_dir="/home/cxx/AI-Agents/AGrail4Agent/DAS/checkpoints/MiniLM-L6-v2")

# print("模型下载完成")


# from sentence_transformers import SentenceTransformer

# # 指定本地模型路径
# model = SentenceTransformer('/home/cxx/AI-Agents/AGrail4Agent/DAS/MiniLM-L6-v2', local_files_only=True)

# # 测试模型是否加载成功
# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(sentences)
# print(embeddings)


# generate an analysis table on csv type

# import csv
# import re
# import json
# import pandas as pd

# # ========== 1. 读取日志文件和 JSON 任务文件 ==========
# with open("/home/cxx/AI-Agents/AGrail4Agent/DAS/outputs/record_benign.txt", "r", encoding="utf-8") as f:
#     full_text = f.read()

# with open("/home/cxx/AI-Agents/AGrail4Agent/DAS/data/safe-os/benign.json", "r") as f:
#     benign_tasks = json.load(f)

# # ========== 2. 分割日志为多个子块，每条记录以 user indentity 开头，以 flag 结尾 ==========
# entry_blocks = re.findall(r"user indentity:.*?flag\s+[01]", full_text, re.DOTALL)

# # ========== 3. 抽取分析字段函数 ==========
# def extract_risk_analysis(block):
#     match = re.search(r"### Analysis of Potential Risks(.*?)(?:\n-{3,}|\n#)", block, re.DOTALL)
#     return match.group(1).lstrip(":").strip() if match else None

# def extract_executor_json(block):
#     match = re.search(r"############Executor###########\s*```json\s*(\[.*?\])\s*```", block, re.DOTALL)
#     if match:
#         try:
#             return json.loads(match.group(1).strip())
#         except json.JSONDecodeError:
#             return match.group(1).strip()
#     return None

# # ========== 4. 提取日志字段 ==========
# structured_entries = []
# for block in entry_blocks:
#     user = re.search(r"user indentity:\s*(\w+)", block)
#     agent = re.search(r"agent indentity:\s*(\w+)", block)
#     command = re.search(r"'action':\s*'bash',\s*'content':\s*'([^']+)'", block)
#     thought = re.search(r"'thought':\s*'([^']+)'", block)
#     flag_match = re.search(r"flag\s+(\d)", block)
#     in_memory = re.search(r"In Memory:\s*(True|False)", block)

#     structured_entries.append({
#         "User": user.group(1) if user else None,
#         "Agent": agent.group(1) if agent else None,
#         "Command": command.group(1) if command else None,
#         "Thought": thought.group(1) if thought else None,
#         "Risk Analysis": extract_risk_analysis(block),
#         "Executor Checks": extract_executor_json(block),
#         "Memory Verdict": in_memory.group(1) if in_memory else None,
#         "Flag": "Failed" if flag_match and flag_match.group(1) == "0" else "Passed"
#     })

# df_os = pd.DataFrame(structured_entries)
# df_os["Index"] = df_os.index

# # ========== 5. 解析 benign.json 元数据并对齐 ==========
# df_meta = pd.DataFrame([
#     {
#         "Description": task.get("description"),
#         "Init Code": task.get("create", {}).get("init", {}).get("code")
#         if isinstance(task.get("create", {}).get("init"), dict)
#         else task.get("create", {}).get("init"),
#         "Evaluation Match": task.get("evaluation", {}).get("match"),
#         "Index": i
#     }
#     for i, task in enumerate(benign_tasks)
# ])

# # ========== 6. 合并分析结果和任务描述 ==========
# df_merged = df_os.merge(df_meta, on="Index", how="left").drop(columns=["Index"])
# df_clean_csv = df_merged.copy()

# # Convert complex JSON (Executor Checks) into compact one-line strings
# df_clean_csv["Executor Checks"] = df_clean_csv["Executor Checks"].apply(
#     lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
# )

# # Export to a LibreOffice-safe CSV with quoting and no embedded newlines
# clean_csv_path = "/home/cxx/AI-Agents/AGrail4Agent/DAS/outputs/analysis/benign_analysis_27_clean.csv"
# df_clean_csv.to_csv(clean_csv_path, index=False, quoting=csv.QUOTE_ALL)

# # ========== 7. 保存或使用结果 ==========
# # output = "/home/cxx/AI-Agents/AGrail4Agent/DAS/outputs/analysis/benign_analysis_27.csv"
# # df_merged.to_csv(output, index=False)
# print(f"save to {clean_csv_path} successfully.")



import json
def check_data_number(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return len(data)

def check_data_number_list(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
        return len(data)

pa_list = [ # safe-os
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/safe-os/benign.json",# 27
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/safe-os/prompt_injection.json", # 23
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/safe-os/system_sabotage.json", # 30
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/safe-os/environment.json", # 20 
    # other datasets
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/Mind2Web_SC/mind2web_sc.json",# Mind2Web_SC 200
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/EICU_AC/eicu_ac.json",# EICU_AC 316
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/AdvWeb/advweb.json",# AdvWeb 12
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/EIA/all_instances.jsonl", # EIA task 177
    "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/EIA/target_element_positions_fixed.json", # EIA target 179
]


pa = pa_list[8]
print(check_data_number(pa))

# pa = pa_list[7]
# print(check_data_number_list(pa))



import json
import jsonlines

# 路径
positions_path = "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/EIA/target_element_positions_fixed.json"
instances_path = "/home/cxx/AI-Agents/AGrail4Agent/DAS/data/EIA/all_instances.jsonl"

# 加载 target 文件为 set of (annotation_id, action_id)
with open(positions_path, "r") as f:
    positions = json.load(f)

positions_pairs = {
    (item["annotation_id"], item["action_id"]): item for item in positions
}

# 加载 task 文件的所有 (annotation_id, action_id)
valid_pairs = set()
with jsonlines.open(instances_path, "r") as reader:
    for obj in reader:
        valid_pairs.add((obj["annotation_id"], obj["action_id"]))

# 找出 positions 中多出来的项
positions_all_pairs = set(positions_pairs.keys())
redundant_pairs = positions_all_pairs - valid_pairs

# 过滤掉冗余项
filtered_positions = [
    item for key, item in positions_pairs.items() if key in valid_pairs
]

print(f"原始定位信息条数: {len(positions)}")
print(f"有效任务中匹配的条数: {len(filtered_positions)}")
print(f"被删除的冗余条目数: {len(redundant_pairs)}")
print(redundant_pairs)

# 保存为新文件
# with open("/home/cxx/AI-Agents/AGrail4Agent/DAS/data/EIA/target_element_positions_aligned.json", "w") as f:
#     json.dump(filtered_positions, f, indent=2)
