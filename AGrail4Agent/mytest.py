# 下载模型文件
from huggingface_hub import snapshot_download

snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model", local_dir="/home/cxx/AI-Agents/AGrail4Agent/DAS/checkpoints/MiniLM-L6-v2")

print("模型下载完成")


# from sentence_transformers import SentenceTransformer

# # 指定本地模型路径
# model = SentenceTransformer('/home/cxx/AI-Agents/AGrail4Agent/DAS/MiniLM-L6-v2', local_files_only=True)

# # 测试模型是否加载成功
# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(sentences)
# print(embeddings)

