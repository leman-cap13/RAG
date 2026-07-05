from pathlib import Path
from rag.chunker import chunk_text
from rag.embedder import embed_text
from rag.vector_store import add_chunks, query
from rag.generator import generate_answer

for file in Path("data").glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    embeddings = [embed_text(c) for c in chunks]
    add_chunks(chunks, embeddings, source=file.name)
    print(f"{file.name}: {len(chunks)} chunk")

while True:
    question = input("\question (output: q): ")
    if question.lower() == "q":
        break
    qv = embed_text(question)
    context = query(qv, top_k=4)
    print("\n" + generate_answer(question, context))

"""
test
"""
