import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai.errors import ClientError # 429
from config import settings


load_dotenv()

api_key = settings.gemini_api_key 

if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY is not defined in your \033[31m.env\033[0m file."
    )

client = genai.Client(api_key=api_key)

MODEL = settings.embed_model
BATCH_SIZE = settings.embed_batch_size


def _embed_batch(texts, task_type):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        while True:
            try:
                response = client.models.embed_content(
                    model=MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                time.sleep(2)
                break
            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("Rate limited. Sleeping 40 seconds...")
                    time.sleep(40)
                    continue
                raise
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
        print(f"-†- chunk {i} -†-")
        print(f"text: {chunk[:50]}...")
        print(f"dim={len(vec)} first5={[round(v, 4) for v in vec[:5]]}\n")
