# add_doc.py
import os
import glob
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import charset_normalizer

# 配置
DATA_DIR = "data/references"
DB_DIR = "db"
COLLECTION_NAME = "documents"

def read_text_file(file_path):
    """自动检测编码并返回文本内容"""
    with open(file_path, "rb") as f:
        raw_data = f.read()
    result = charset_normalizer.detect(raw_data)
    encoding = result["encoding"]
    if encoding is None:
        encoding = "utf-8"
    try:
        return raw_data.decode(encoding)
    except (UnicodeDecodeError, TypeError):
        return raw_data.decode("utf-8", errors="ignore")

def main():
    # 检查数据库是否存在
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("❌ Database not found. Please run 'init_db.py' first.")
        return

    # 获取要添加的文件列表
    if len(sys.argv) > 1:
        # 支持传入具体文件路径，如: python add_doc.py data/new1.txt data/new2.txt
        file_paths = [f for f in sys.argv[1:] if f.endswith('.txt') and os.path.isfile(f)]
    else:
        # 默认：添加 data/ 下所有 .txt（包括之前可能没加的）
        file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))

    if not file_paths:
        print(f"⚠️ No .txt files to add.")
        return

    # 加载模型和客户端
    print("Loading embedding model...")
    # ✅ 使用本地模型（离线）
    model = SentenceTransformer('./models/bge-small-zh-v1.5')
    client = QdrantClient(path=DB_DIR)

    # 检查集合是否存在
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' not found. Did you run init_db.py?")
        return

    points = []
    # 获取当前最大 ID（用于新 point 的 ID）
    # Qdrant 不提供直接获取 max_id 的方法，我们用一个简单策略：从现有点数估算
    # 更严谨的做法是维护一个外部计数器，但为简化，我们用时间戳或大基数 ID
    import time
    base_id = int(time.time() * 1000)  # 用毫秒时间戳作为起始 ID，避免冲突

    point_id = base_id
    added_files = []

    for file_path in file_paths:
        print(f"Adding {file_path}...")
        try:
            content = read_text_file(file_path)
        except Exception as e:
            print(f"⚠️ Skip {file_path}: {e}")
            continue

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [content] if content.strip() else []

        for para in paragraphs:
            if not para:
                continue
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
        added_files.append(os.path.basename(file_path))

    if points:
        print(f"Inserting {len(points)} new vectors into Qdrant...")
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Successfully added files: {', '.join(added_files)}")
    else:
        print("⚠️ No valid content to add.")

if __name__ == "__main__":
    main()