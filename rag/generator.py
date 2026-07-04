import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

PROMPT_TEMPLATE = """Aşağıdakı məlumata əsasən suala cavab ver.

Məlumat:
{context}

Sual: {question}

Cavab:"""

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text