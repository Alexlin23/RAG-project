# api_query.py
"""
本地 RAG 查询 HTTP API 服务
- 使用 FastAPI 提供 RESTful 接口
- 调用本地 Qdrant 向量数据库进行语义检索
- 支持中文查询
- 默认监听 http://localhost:8000

📌 API 文档（自动生成）：
    - Swagger UI: http://localhost:8000/docs
    - ReDoc:        http://localhost:8000/redoc

📦 依赖安装：
    pip install fastapi uvicorn qdrant-client sentence-transformers charset-normalizer

🚀 启动服务：
    python api_query.py

📝 示例请求：
    POST /query
    {
        "text": "人工智能是什么？",
        "top_k": 3
    }


    curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "人工智能是什么？",
    "top_k": 2
  }'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ----------------------------
# 配置常量
# ----------------------------
DB_DIR = "db"
COLLECTION_NAME = "documents"
DEFAULT_TOP_K = 3
MAX_TOP_K = 10  # 防止用户请求过大结果集

# ----------------------------
# 初始化模型与数据库客户端（启动时加载一次）
# ----------------------------
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    raise RuntimeError("❌ 数据库未初始化！请先运行 init_db.py")

print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer('./models/bge-small-zh-v1.5')
QDRANT_CLIENT = QdrantClient(path=DB_DIR)

if not QDRANT_CLIENT.collection_exists(COLLECTION_NAME):
    raise RuntimeError(f"❌ 集合 '{COLLECTION_NAME}' 不存在，请先运行 init_db.py")

# ----------------------------
# FastAPI 应用
# ----------------------------
app = FastAPI(
    title="Local RAG Query API",
    description="基于本地向量数据库的语义检索服务，无需联网，支持中文。",
    version="1.0.0"
)

# ----------------------------
# 请求/响应数据模型
# ----------------------------
class QueryRequest(BaseModel):
    text: str                      # 用户查询文本
    top_k: Optional[int] = None   # 返回结果数量（可选，默认3）

class SearchResultItem(BaseModel):
    score: float                  # 相似度分数（余弦相似度，范围 [-1, 1]）
    text: str                     # 检索到的原文片段
    source_file: str              # 来源文件名

class QueryResponse(BaseModel):
    query: str                    # 原始查询
    results: List[SearchResultItem]  # 检索结果列表
    total: int                    # 结果总数

# ----------------------------
# API 路由
# ----------------------------
@app.post("/query", response_model=QueryResponse, summary="执行语义检索")
async def query_endpoint(request: QueryRequest):
    """
    根据用户输入的自然语言问题，在本地文档库中检索最相关的文本片段。

    **参数说明**:
    - `text`: 必填，要查询的问题（支持中文）
    - `top_k`: 可选，返回结果数量（默认 3，最大 10）

    **返回示例**:
    ```json
    {
        "query": "深度学习是什么？",
        "results": [
            {
                "score": 0.8721,
                "text": "深度学习是机器学习的子集。",
                "source_file": "doc2.txt"
            }
        ],
        "total": 1
    }
    ```
    """
    query_text = request.text.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    # 处理 top_k
    top_k = request.top_k or DEFAULT_TOP_K
    if top_k < 1:
        top_k = 1
    if top_k > MAX_TOP_K:
        top_k = MAX_TOP_K

    try:
        # 1. 将查询文本向量化
        query_vector = EMBEDDING_MODEL.encode(query_text).tolist()

        # 2. 在 Qdrant 中执行向量搜索
        search_result = QDRANT_CLIENT.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )

        # 3. 构造响应结果
        results = []
        for hit in search_result.points:
            results.append(
                SearchResultItem(
                    score=round(hit.score, 4),
                    text=hit.payload.get("text", ""),
                    source_file=hit.payload.get("source_file", "unknown")
                )
            )

        return QueryResponse(
            query=query_text,
            results=results,
            total=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


# ----------------------------
# 健康检查接口
# ----------------------------
@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务是否正常运行"""
    return {"status": "ok", "model": "BAAI/bge-small-zh-v1.5", "collection": COLLECTION_NAME}


# ----------------------------
# 启动入口（用于直接运行）
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 RAG 查询 API 服务...")
    print("📖 访问 http://localhost:8000/docs 查看交互式 API 文档")
    uvicorn.run(app, host="0.0.0.0", port=8000)