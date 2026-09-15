import os
from typing import Any

from fastembed import TextEmbedding
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from qdrant_client import QdrantClient

DB_PATH = os.getenv("QDRANT_DB_PATH", "./qdrant_db")
COLLECTION_NAME = "adk_docs"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

qdrant_client = QdrantClient(path=DB_PATH)
embedding_model = TextEmbedding(model_name=MODEL_NAME)


def query_qdrant_docs(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Retrieves relevant documentation chunks from the local Qdrant vector database.

    Args:
        query: The natural language search query or concept to search for.
        limit: Maximum number of relevant documentation passages to return. Defaults to 3.

    Returns:
        A list of dictionaries containing payload metadata and matched chunk content.
    """
    # Query Qdrant using built-in vector search/payload filtering
    #
    query_vector = next(iter(embedding_model.embed([query]))).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    documents: list[dict[str, Any]] = []
    for point in results.points:
        payload = point.payload or {}
        documents.append(
            {
                "score": point.score,
                "path": payload.get("path", "unknown"),
                "text": payload.get("text", ""),
                "content_type": payload.get("content", ""),
            }
        )

    return documents


LLM_MODEL_NAME = "gemini-3.5-flash-lite"

qdrant_tool = FunctionTool(query_qdrant_docs)
root_agent = LlmAgent(
    name="adk_documentation_agent",
    model=LLM_MODEL_NAME,
    instruction=(
        "You are an expert technical assistant for the Agent Development Kit (ADK). "
        "Use the `query_qdrant_docs` tool whenever you need accurate, grounding "
        "information from the local documentation base to answer user queries."
    ),
    tools=[qdrant_tool],
)
