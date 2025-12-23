# init_db.py
import os
import glob
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# 配置路径
DATA_DIR = "data/references"
DB_DIR = "db"
COLLECTION_NAME = "documents"

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 初始化 embedding 模型（中文推荐）
print("Loading embedding model...")
# ✅ 使用本地模型（离线）
model = SentenceTransformer('./models/bge-small-zh-v1.5')  # 中文效果好
# 如果你处理英文，可改用：'sentence-transformers/all-MiniLM-L6-v2'

# 获取向量维度
test_emb = model.encode("测试")
VECTOR_SIZE = len(test_emb)

# 启动本地 Qdrant（持久化到 db/ 目录）
print("Starting Qdrant client...")
client = QdrantClient(path=DB_DIR)  # 使用本地存储模式

# 创建集合（如果不存在）
if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
else:
    print(f"Collection {COLLECTION_NAME} already exists. Clearing it for re-initialization.")
    client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# 读取所有 txt 文件
txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
if not txt_files:
    print(f"⚠️ No .txt files found in {DATA_DIR}/")
    exit()

points = []
point_id = 1

for file_path in txt_files:
    print(f"Processing {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # 简单按空行或段落分割（你可以后续改成更智能的分块）
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    if not paragraphs:
        paragraphs = [content]  # 如果没分段，整篇作为一块
    
    for para in paragraphs:
        emb = model.encode(para).tolist()
        points.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "text": para,
                    "source_file": os.path.basename(file_path)
                }
            )
        )
        point_id += 1

# 批量插入
print(f"Inserting {len(points)} vectors into Qdrant...")
client.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Database initialized successfully! Data stored in 'db/' folder.")