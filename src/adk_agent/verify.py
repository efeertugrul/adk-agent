from qdrant_client import QdrantClient

if __name__ == "__main__":
    client = QdrantClient(path="./qdrant_db")
    info = client.get_collection("adk_docs")
    print(f"Points count: {info.points_count} | Status: {info.status}")
