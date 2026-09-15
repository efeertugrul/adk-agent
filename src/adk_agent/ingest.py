import os
from collections.abc import Iterator

import httpx
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from adk_agent.rate_limitter import RateLimitTracker

OWNER = "google"
REPO = "adk-python"
BRANCH = "main"

tree_url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/docs?ref={BRANCH}"
headers = {"User-Agent": "adk-agent"}

COLLECTION_NAME = "adk_docs"
QDRANT_PATH = "./qdrant_db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # The embedding vector size for the model

if token := os.getenv("GITHUB_TOKEN"):
    headers["Authorization"] = f"Bearer {token}"


def fetch_markdown_files(
    dir_url: str, client: httpx.Client, tracker: RateLimitTracker
) -> Iterator[tuple[str, str]]:
    """Recursively fetches .md docs from GitHub while managing rate limits and retries."""
    while True:
        tracker.wait_if_needed()

        response = client.get(dir_url)
        tracker.update(response.headers)
        tracker.display_status()

        if response.status_code == 403 and tracker.remaining == 0:
            print("\n[ERROR] GitHub API rate limit reached! Ingestion paused.")
            tracker.wait_if_needed()
            continue

        response.raise_for_status()
        files_data = response.json()
        break

    for item in files_data:
        item_type = item.get("type")
        path = item.get("path")

        if item_type == "dir":
            yield from fetch_markdown_files(item.get("url"), client, tracker)
        elif item_type == "file" and path.endswith(".md"):
            raw_url = item.get("download_url")
            if raw_url:
                doc_response = client.get(raw_url)
                if doc_response.status_code == 200:
                    print(f"Downloading the file: {path} into RAM...")
                    yield path, doc_response.text


def build_index():
    """Generates embeddings locally and indexes chunks into Qdrant."""
    token = os.getenv("GITHUB_TOKEN")
    tracker = RateLimitTracker(token=token)
    headers = {"User-Agent": "adk-rag-assistant"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    embedding_model = TextEmbedding(model_name=MODEL_NAME)
    qdrant = QdrantClient(path=QDRANT_PATH)

    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  ## model has a max token limit of 512 tokens so we set chunk size to 800 characters to be safe.
        chunk_overlap=150,  ## to make sure we capture the context of the text
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    point_id = 0

    print("Starting streaming ingestion from GitHub...")
    with httpx.Client(follow_redirects=True, headers=headers) as client:
        for file_path, content in fetch_markdown_files(tree_url, client, tracker):
            print(
                f"Processing: {file_path} (Quota left: {tracker.remaining}/{tracker.limit})"
            )
            chunks = text_splitter.split_text(content)
            if not chunks:
                print(f"No chunks generated for {file_path}. Skipping.")
                continue

            embeddings = list(embedding_model.embed(chunks))

            points = []
            for chunk, vector in zip(chunks, embeddings):
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={"source": file_path, "text": chunk},
                    )
                )
                point_id += 1
            if points:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Ingestion complete. Total chunks indexed: {point_id}")


if __name__ == "__main__":
    build_index()
