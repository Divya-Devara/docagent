"""DocAgent web API: receipt OCR text in, validated receipt JSON out."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docagent.extract import extract
from docagent.schema import Receipt

app = FastAPI(title="DocAgent", description="Extracts company, date, address and total from receipt OCR text.")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="Receipt OCR text, one line per row")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=Receipt)
def extract_receipt(request: ExtractRequest) -> Receipt:
    try:
        receipt = extract(request.text)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="LLM provider is unavailable")
    if receipt is None:
        raise HTTPException(status_code=422, detail="Could not extract a valid receipt after 3 attempts")
    return receipt