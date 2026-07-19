import logging
import time

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

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
        logger.warning("generate_answer_no_context")
        return "Təəssüf ki, bu barədə mənbələrdə məlumat tapa bilmədim."

    context = _format_sources(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=settings.gen_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature if temperature is not None else settings.gen_temperature
        ),
    )

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.debug(
        "generate_answer",
        extra={"model": settings.gen_model, "context_chunks": len(context_chunks), "duration_ms": duration_ms},
    )
    return response.text