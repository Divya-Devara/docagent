# DocAgent: receipt field extraction with measured accuracy

**Status: In progress.** LLM extraction, evaluation, PyTorch/ONNX classifier and A/B/A+B comparison done. Next: FastAPI + Docker + deployment.

DocAgent extracts **company, date, address and total** from receipt OCR text, validates the output, and measures its own accuracy against ground truth. It compares three approaches: an LLM, a small ML classifier, and a hybrid of both.

## Problem

Finance teams type receipt fields into their systems by hand. That's slow and error-prone. DocAgent extracts the fields automatically, and every claim about accuracy, speed and size below is measured, not assumed.

## Architecture

    OCR text -> Loader -> A: LLM extractor (Pydantic schema, validation, retry with error feedback)
                       -> B: Line classifier (PyTorch MLP -> ONNX -> ONNX Runtime, INT8)
                       -> A+B: classifier selects lines, LLM reads only those
             -> Normalizer -> Evaluator (field-level exact + fuzzy accuracy, latency)

| Component | File | Job |
|---|---|---|
| Loader | `docagent/loader.py` | Reads OCR lines and ground truth from SROIE |
| Schema | `docagent/schema.py` | Pydantic `Receipt`: the output contract |
| A: LLM extractor | `docagent/extract.py` | Ollama, JSON-schema output, temperature 0, up to 3 attempts with the validation error fed back |
| Labeller | `docagent/labels.py` | Auto-labels every OCR line from the ground truth |
| Features | `docagent/features.py` | Character n-grams of the line and the previous line; receipt-level train/val split |
| B: Classifier | `docagent/classifier.py`, `train.py` | MLP with class-weighted loss, model chosen by validation macro-F1 |
| ONNX | `export_onnx.py`, `quantize.py` | Export, parity check, latency, dynamic INT8 quantization |
| B: Field rules | `docagent/line_extract.py` | Turns line labels into fields (most confident line, date regex, joined address lines) |
| A+B: Hybrid | `docagent/hybrid.py` | Header lines + candidate lines and their neighbours sent to the LLM |
| Evaluator | `eval.py` | Runs any extractor on any split; writes a report with every mistake |

## Data

[SROIE dataset v2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) (ICDAR 2019): 626 train and 347 test scanned receipts, each with OCR lines and ground truth. A data audit (in `tests/test_loader.py`) found one train receipt with no address in its ground truth; the evaluator skips missing fields instead of counting them as right or wrong.

## Results

### 1. Prompt development (first 50 train receipts)

| Field | v1 exact | v2 exact | v2 fuzzy |
|---|---|---|---|
| company | 80% | 82% | 88% |
| date | 86% | 96% | 100% |
| address | 34% | 42% | 96% |
| total | 96% | 96% | 96% |

Error analysis of v1 showed that all 7 date errors were correct dates rewritten as YYYY-MM-DD. One prompt rule ("copy the date exactly as printed") fixed them. With 50 receipts each receipt is 2%, so the small company change is within noise.

### 2. Comparison on the held-out test set (first 50 test receipts)

Same receipts, same normalization and scoring (fuzzy = difflib ratio >= 0.9). Nothing was tuned on these receipts.

| Field | A: LLM only | B: Classifier only | A+B: Hybrid |
|---|---|---|---|
| company | 80% | 66% | 80% |
| date | 94% | 96% | 94% |
| address | 94% | 84% | 90% |
| total | 94% | 76% | 72% |
| Latency p50 | 5,467 ms | 1.1 ms | 4,720 ms |
| Model size | ~4.7 GB | 256 KB (+50 KB vectorizers) | both |

- **A:** qwen2.5:7b via Ollama on a Mac, 0 extraction failures.
- **B:** trained on 33,626 auto-labelled lines (87% "other"); validation macro-F1 0.856. Weakest class: total (F1 0.552, precision 0.399), because other numbers such as cash paid look like totals.
- **A+B:** the classifier selects XX% of the text for the LLM.

### 3. ONNX Runtime and INT8 quantization (classifier, validation set)

| | FP32 ONNX | INT8 ONNX |
|---|---|---|
| Size | 1,008 KB | 256 KB (3.9x smaller) |
| Macro-F1 | 0.856 | 0.856 |
| Latency, 50 lines | ~0.04-0.06 ms | ~0.05 ms (within noise) |

PyTorch-to-ONNX parity: identical predictions, max output difference 3e-6.

## Findings

- **The LLM is the most accurate** on total, company and address; the **classifier is ~5,000x faster** and best on dates.
- **The hybrid did not pay off:** only 14% faster, because generating the output, not reading the input, dominates LLM latency. Total accuracy dropped from 94% to 72%, because the removed lines held the context that separates the total from cash and subtotal lines. Deployed choice: LLM only.
- **Validation loss and F1 disagreed:** the lowest validation loss was at epoch 3 (macro-F1 0.820), the best macro-F1 at epoch 9 (0.856). The model is selected by macro-F1.
- **INT8 reduced size, not latency,** because ~95% of the classifier's 1.1 ms is feature extraction, not the network.

## Known limits

- 50 receipts per evaluation, so each receipt is 2 percentage points.
- Some ground-truth company labels are ambiguous (brand name vs legal company name).
- The line labeller found a company line in only 514 of 626 train receipts (company names split across OCR lines), which limits the classifier on that field.
- Input is OCR text, not images.

## Run it

    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # Download "SROIE dataset v2" from Kaggle into data/sroie/{train,test}/{box,entities}
    ollama pull qwen2.5:7b

    python -m pytest -v                                         # tests
    python eval.py results/eval_v2.md llm train                 # LLM on the dev set
    python train.py && python export_onnx.py && python quantize.py   # classifier
    python eval.py results/test_classifier.md classifier test   # classifier on test
    python eval.py results/test_llm.md llm test                 # LLM on test
    python eval.py results/test_hybrid.md hybrid test           # hybrid on test

## Next

FastAPI service (`POST /extract`, `GET /health`), LLM provider switch (Ollama locally, Gemini when deployed), Docker, and deployment to Hugging Face Spaces.