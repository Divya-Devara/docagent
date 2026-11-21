"""Extractor A: asks the LLM for a Receipt and checks the answer."""
import ollama
from pydantic import ValidationError

from docagent.schema import Receipt

MODEL = "qwen2.5:7b"
MAX_ATTEMPTS = 3
PROMPT = """Extract these fields from the receipt OCR text below.
Return JSON only, with these rules:
- company: the seller's business name
- date: the purchase date copied exactly as printed (same format, e.g. 25/12/2018 or 03 MAR 2018); do not convert it; no time
- address: the seller's full address, all lines joined with single spaces
- total: the final amount paid, as a number without currency

Receipt:
{text}"""
def extract(text: str) -> Receipt | None:
    messages = [{"role": "user", "content": PROMPT.format(text=text)}]

    for attempt in range(MAX_ATTEMPTS):
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            format=Receipt.model_json_schema(),
            options={"temperature": 0},
        )
        answer = response["message"]["content"]

        try:
            return Receipt.model_validate_json(answer)
        except ValidationError as error:
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": f"Your answer was invalid:\n{error}\nReturn corrected JSON only."})

    return None