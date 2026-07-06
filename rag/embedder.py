import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL = "gemini-embedding-001"
BATCH_SIZE = 100


def _embed_batch(texts, task_type):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.models.embed_content(
            model=MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.extend(e.values for e in response.embeddings)
    return embeddings


def embed_documents(texts):
    return _embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text):
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag.chunker import chunk_text

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = next(Path("data").glob("*.txt"))

    text = open(path, encoding="utf-8").read()
    chunks = chunk_text(text)
    print(f"{path}: {len(chunks)} chunks to embed\n")

    embeddings = embed_documents(chunks)
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        print(f"--- chunk {i} ---")
        print(f"text: {chunk[:80]}...")
        print(f"dim={len(vec)} first5={[round(v, 4) for v in vec[:5]]}\n")
