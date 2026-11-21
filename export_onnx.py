"""Exports the classifier to ONNX, checks it matches PyTorch, and measures latency."""
import time

import numpy as np
import onnxruntime as ort
import torch

from docagent.classifier import LineClassifier

N_FEATURES = 2000
ONNX_PATH = "models/line_classifier.onnx"

# 1. Load the trained PyTorch model
model = LineClassifier(N_FEATURES)
model.load_state_dict(torch.load("models/line_classifier.pt"))
model.eval()

# 2. Export to ONNX
dummy = torch.zeros(1, N_FEATURES)
torch.onnx.export(
    model, (dummy,), ONNX_PATH,
    input_names=["features"], output_names=["scores"],
    dynamic_axes={"features": {0: "batch"}, "scores": {0: "batch"}},
)

# 3. Parity: same input, compare outputs
rng = np.random.default_rng(42)
x = (rng.random((500, N_FEATURES)) < 0.02).astype(np.float32)
with torch.no_grad():
    torch_scores = model(torch.from_numpy(x)).numpy()
session = ort.InferenceSession(ONNX_PATH)
onnx_scores = session.run(None, {"features": x})[0]
print("max difference:", float(np.abs(torch_scores - onnx_scores).max()))
print("same predictions:", bool((torch_scores.argmax(1) == onnx_scores.argmax(1)).all()))


# 4. Latency for one receipt (about 50 lines at once)
def measure(run, runs: int = 1000) -> tuple[float, float]:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[runs // 2], times[int(runs * 0.95)]


receipt = x[:50]
receipt_tensor = torch.from_numpy(receipt)
with torch.no_grad():
    torch_p50, torch_p95 = measure(lambda: model(receipt_tensor))
onnx_p50, onnx_p95 = measure(lambda: session.run(None, {"features": receipt}))
print(f"PyTorch:      p50 {torch_p50:.3f} ms  p95 {torch_p95:.3f} ms")
print(f"ONNX Runtime: p50 {onnx_p50:.3f} ms  p95 {onnx_p95:.3f} ms")