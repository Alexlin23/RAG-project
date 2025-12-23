# query.py （修正版）
import os
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

DB_DIR = "db"
COLLECTION_NAME = "documents"
TOP_K = 10

def main():
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("❌ Database not found. Please run 'init_db.py' first.")
        return

    print("Loading embedding model...")
    # ✅ 使用本地模型（离线）
    model = SentenceTransformer('./models/bge-small-zh-v1.5')
    client = QdrantClient(path=DB_DIR)

    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("请输入你的问题: ").strip()
    if not query_text:
        print("⚠️ 查询内容为空。")
        return

    query_vector = model.encode(query_text).tolist()

    try:
        # ✅ 使用 query_points 替代旧的 search
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,      # 注意：参数名是 query，不是 query_vector
            limit=TOP_K
        ).points  # 返回的是 SearchResult 对象，需取 .points
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return

    if not results:
        print("📭 没有找到相关文档。")
        return

    print(f"\n🔍 找到 {len(results)} 个相关片段（Top-{TOP_K}）:\n")
    for i, hit in enumerate(results, 1):
        score = hit.score
        text = hit.payload.get("text", "")
        source = hit.payload.get("source_file", "unknown")
        print(f"{i}. 相似度: {score:.4f} | 来源: {source}")
        print(f"   内容: {text}\n")

if __name__ == "__main__":
    main()