import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

client = genai.Client(api_key=settings.gemini_api_key)

PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8")


def _format_sources(context_chunks):
    lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source") or "naməlum"
        lines.append(f"[{i}] (mənbə: {source})\n{chunk['text']}")
    return "\n\n".join(lines)


def _format_history(history):
    if not history:
        return ""
    lines = ["\nƏvvəlki söhbət:"]
    for turn in history:
        speaker = "Tələbə" if turn.get("role") == "user" else "Köməkçi"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines) + "\n"


def _greeting_rule(history):
    if history:
        return "Bu, söhbətin davamıdır — YENİDƏN salamlaşma, 'necəsən' demə, birbaşa suala cavab ver."
    return "Bu, söhbətin ilk mesajıdır — istəsən qısa və təbii şəkildə salamlaya bilərsən."


def _total_sources_note(total_sources):
    if not total_sources:
        return ""
    return f"Bazada ümumilikdə {total_sources} universitet var."


def generate_answer(question, context_chunks, history=None, temperature=None, total_sources=None):
    if not context_chunks:
        logger.warning("generate_answer_no_context")
        return "Təəssüf ki, bu barədə mənbələrdə məlumat tapa bilmədim."

    context = _format_sources(context_chunks)
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        history=_format_history(history),
        greeting_rule=_greeting_rule(history),
        total_sources_note=_total_sources_note(total_sources),
    )

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