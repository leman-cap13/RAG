from config import settings 
from rag.embedder import embed_query
from rag.vector_store import query
from rag.generator import generate_answer
from rag.ingest import index_all

for result in index_all():
    if result["status"] == "skipped":
        print(f"{result['file']}: already indexed, skipping")
    else:
        print(f"{result['file']}: {result['chunks']} chunks indexed")

while True:
    question = input("\nquestion (type q to quit): ")
    if question.lower() == "q":
        break

    qv = embed_query(question)
    context = query(qv, top_k=4)

    print(f"\n{len(context)} chunks retrieved from the vector store:")
    for i, c in enumerate(context, start=1):
        print(f"  [{i}] source={c['source']} distance={c['distance']:.4f}")
        print(f"      {c['text']}")

    print("\n--- LLM answer ---")
    print(generate_answer(question, context))

    sources = sorted({c["source"] for c in context if c.get("source")})
    if sources:
        print("\nSources: " + ", ".join(sources))
