# DocAgent: receipt field extraction with measured accuracy

**Status: In progress.** Session 1 (LLM extraction + evaluation) done. Next: PyTorch/ONNX classifier, FastAPI, deployment.

Extracts **company, date, address and total** from receipt OCR text, validates the output, and measures its own accuracy against ground truth.

## How it works
OCR text -> Loader -> LLM extractor (qwen2.5:7b via Ollama) -> Pydantic validation (retry with error feedback, max 3) -> Normalizer -> Evaluator

- Schema-constrained JSON output, temperature 0
- Deterministic validation before trusting the LLM
- Field-level exact and fuzzy (difflib >= 0.9) accuracy

## Results (50 SROIE train receipts, development set)

| Field | v1 exact | v2 exact | v2 fuzzy |
|---|---|---|---|
| company | 80% | 82% | 88% |
| date | 86% | 96% | 100% |
| address | 34% | 42% | 96% |
| total | 96% | 96% | 96% |

Extraction failures: 0/50. Latency: p50 5.4 s, p95 6.6 s per receipt (local CPU/GPU, Mac).

**v1 -> v2:** error analysis showed all 7 date errors were correct dates reformatted to YYYY-MM-DD. One prompt rule ("copy the date exactly as printed") fixed them.

**Known limits:** some company labels in the ground truth are ambiguous (brand vs legal name). With 50 receipts, each receipt is 2%. Final results will be reported on the held-out test split.

## Run it
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # Download "SROIE dataset v2" from Kaggle into data/sroie/{train,test}/{box,entities}
    ollama pull qwen2.5:7b
    python -m pytest -v
    python eval.py results/eval_v1.md