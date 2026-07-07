import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY is required in the environment to generate answers.")

client = genai.Client(api_key=api_key)

MODEL = "gemini-2.5-flash"
MIN_SIMILARITY = 0.25
PROMPT_TEMPLATE = """Sən tələbələrə universitet seçimində kömək edən səmimi, isti münasibətli bir köməkçisən.

Qaydalar:
- Cavabı YALNIZ aşağıdakı mənbələrdəki məlumata əsasən ver.
- Rəsmi, quru dillə yox, tələbə ilə söhbət edir kimi mehriban və anlaşıqlı tonda yaz.
- Əgər mənbələrdə sualın cavabını etibarlı şəkildə mənbələrdən çıxara bilməsən, bunu səmimi şəkildə bildir.
- Uyğun olduğu yerlərdə istifadə etdiyin mənbəni [1], [2] və s. şəklində qeyd et.
- Cavabı Azərbaycan dilində yaz.

Mənbələr:
{context}

Sual: {question}

Cavab:"""


def _format_sources(context_chunks):
    lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source") or "naməlum"
        lines.append(f"[{i}] (mənbə: {source})\n{chunk['text']}")
    return "\n\n".join(lines)


def _fallback_message():
    return "Təəssüf ki, sualın cavabını mövcud mənbələrdən etibarlı şəkildə çıxara bilmədim."


def _has_enough_similarity(context_chunks, min_similarity):
    for chunk in context_chunks:
        if chunk.get("similarity", 0.0) >= min_similarity:
            return True
    return False


def generate_answer(question, context_chunks, temperature=0.2, min_similarity=MIN_SIMILARITY):
    if not context_chunks:
        return _fallback_message()

    if not _has_enough_similarity(context_chunks, min_similarity):
        return _fallback_message()

    context = _format_sources(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text
