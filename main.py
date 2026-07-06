from pathlib import Path
from rag.chunker import chunk_text
from rag.embedder import embed_documents, embed_query
from rag.vector_store import add_chunks, is_source_indexed, query
from rag.generator import generate_answer

for file in Path("data").glob("*.txt"):
    if is_source_indexed(file.name):
        print(f"{file.name}: already indexed, skipping")
        continue

    text = file.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    embeddings = embed_documents(chunks)
    add_chunks(chunks, embeddings, source=file.name)

    print(f"\n{file.name}: {len(chunks)} chunks indexed")
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"  chunk {i} ({len(chunk)} chars, vector size: {len(embedding)}):")
        print(f"    {chunk}")

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
