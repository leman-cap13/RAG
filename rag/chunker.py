import re
from config import settings
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return sentences


def chunk_text(text, chunk_size=None, overlap=None):
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))

            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_sentences and overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = next(Path("data").glob("*.txt"))


    text = open(path, encoding="utf-8").read()
    chunks = chunk_text(text)

    print(f"{path}: {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"-~- chunk {i} ({len(chunk)} chars) -~-")
        print(chunk)
        print()
 