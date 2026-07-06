import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []

    for paragraph in paragraphs:
        sentences.extend(
            s.strip()
            for s in _SENTENCE_SPLIT.split(paragraph)
            if s.strip()
        )

    return sentences


def chunk_text(text, chunk_size=800, overlap=150):
    sentences = _split_sentences(text)

    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > chunk_size:
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
