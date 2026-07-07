from pathlib import Path

from rag.chunker import chunk_text
from rag.embedder import embed_documents, embed_query
from rag.generator import generate_answer, MIN_SIMILARITY
from rag.vector_store import add_chunks, query, is_source_indexed


def index_data(source_dir="data"):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Warning: data directory '{source_dir}' not found.")
        return

    while True:
        ans = input("would you like to update the vector database? (y/n): ")
        if ans == 'n':
            for file in source_path.glob("*.txt"):
                if is_source_indexed(file.name):
                    print(f"{file.name}: already indexed, skipping")
                    continue

                text = file.read_text(encoding="utf-8")
                chunks = chunk_text(text)
                embeddings = embed_documents(chunks)
                add_chunks(chunks, embeddings, source=file.name)

                print(f"\n{file.name}: {len(chunks)} chunks indexed")
            return
        elif ans == 'y':
            return
        else: continue


def show_context(context):
    print(f"\n{len(context)} chunks retrieved from the vector store:")
    for i, c in enumerate(context, start=1):
        similarity = c.get("similarity")
        score_text = f"distance={c['distance']:.4f}" if c.get("distance") is not None else ""
        if similarity is not None:
            score_text += f" similarity={similarity:.4f}"
        print(f"  [{i}] source={c['source']} {score_text}")
        print(f"      {c['text'][:50]}...")


def ask_loop():
    while True:
        question = input("\nquestion (type q to quit): ")
        if question.lower() == "q":
            break

        qv = embed_query(question)
        context = query(qv, top_k=4)
        show_context(context)

        print("\n--- \033[36manswer\033[0m ---")
        print(generate_answer(question, context))

        sources = sorted({c["source"] for c in context if c.get("source")})
        if sources:
            print("\n\033[34mSources: \033[35m" + ", ".join(sources) + "\033[0m")





if __name__ == "__main__":
    index_data()
    print(f"\nUsing similarity threshold: \033[33m{MIN_SIMILARITY}\033[0m")
    ask_loop()

