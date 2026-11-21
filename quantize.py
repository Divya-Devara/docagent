"""Quantizes the ONNX classifier to INT8 and compares size, speed and accuracy."""
import os
import time

import joblib
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score

from docagent.features import split_by_receipt, to_matrix
from docagent.labels import LABELS, build_dataset

FP32 = "models/line_classifier.onnx"
INT8 = "models/line_classifier_int8.onnx"

# 1. Quantize the weights to INT8
quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)

# 2. Real validation data (same seed, so the same validation receipts as training)
line_vec, prev_vec = joblib.load("models/vectorizers.joblib")
_, val_rows = split_by_receipt(build_dataset("train"))
X_val = to_matrix(val_rows, line_vec, prev_vec)
y_val = np.array([LABELS.index(r["label"]) for r in val_rows])
receipt = X_val[:50]


def latency_ms(session, runs: int = 1000) -> tuple[float, float]:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"features": receipt})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[runs // 2], times[int(runs * 0.95)]


# 3. Compare FP32 vs INT8
for name, path in [("FP32", FP32), ("INT8", INT8)]:
    session = ort.InferenceSession(path)
    predicted = session.run(None, {"features": X_val})[0].argmax(axis=1)
    macro_f1 = f1_score(y_val, predicted, average="macro")
    p50, p95 = latency_ms(session)
    size_kb = os.path.getsize(path) / 1024
    print(f"{name}: size {size_kb:7.1f} KB   macro-F1 {macro_f1:.3f}   p50 {p50:.3f} ms   p95 {p95:.3f} ms")