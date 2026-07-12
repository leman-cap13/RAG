from google import genai
from google.genai import types
from config import settings

client = genai.Client(api_key=settings.gemini_api_key)
MODEL = settings.gen_model

PROMPT_TEMPLATE = """Sən tələbələrə universitet seçimində kömək edən səmimi, isti münasibətli bir köməkçisən.

Qaydalar:
- Cavabı YALNIZ aşağıdakı mənbələrdəki məlumata əsasən ver.
- Rəsmi, quru dillə yox, tələbə ilə söhbət edir kimi mehriban və anlaşıqlı tonda yaz.
- Əgər mənbələrdə sualın cavabı yoxdursa, bunu səmimi şəkildə bildir, məsələn: "Təəssüf ki, bu barədə mənbələrdə məlumat tapa bilmədim."
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


def generate_answer(question, context_chunks, temperature=None):
    if not context_chunks:
        return "Təəssüf ki, bu barədə mənbələrdə məlumat tapa bilmədim."

    context = _format_sources(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature if temperature is not None else settings.gen_temperature),
    )
    return response.text