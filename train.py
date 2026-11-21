"""Trains the line classifier (Extractor B) and reports per-class F1."""
from pathlib import Path
import copy
import joblib
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from docagent.classifier import LineClassifier
from docagent.features import make_vectorizers, split_by_receipt, to_matrix
from docagent.labels import LABELS, build_dataset

torch.manual_seed(42)
EPOCHS = 10
BATCH_SIZE = 256
MODEL_DIR = Path("models")

# 1. Data: labels -> split -> numbers -> tensors
train_rows, val_rows = split_by_receipt(build_dataset("train"))
line_vec, prev_vec = make_vectorizers()
line_vec.fit([r["line"] for r in train_rows])
prev_vec.fit([r["previous"] for r in train_rows])
X_train = torch.from_numpy(to_matrix(train_rows, line_vec, prev_vec))
X_val = torch.from_numpy(to_matrix(val_rows, line_vec, prev_vec))
y_train = torch.tensor([LABELS.index(r["label"]) for r in train_rows])
y_val = torch.tensor([LABELS.index(r["label"]) for r in val_rows])

# 2. Model, weighted loss, optimizer
model = LineClassifier(X_train.shape[1])
counts = torch.bincount(y_train, minlength=len(LABELS)).float()
weights = counts.sum() / (len(LABELS) * counts)
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("class weights:", dict(zip(LABELS, weights.round(decimals=2).tolist())))
best_f1, best_state, best_epoch = 0.0, None, 0
# 3. Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(X_train))
    total_loss = 0.0
    for start in range(0, len(X_train), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        val_scores = model(X_val)
        val_loss = loss_fn(val_scores, y_val).item()
        val_f1 = f1_score(y_val, val_scores.argmax(dim=1), average="macro")
    print(f"epoch {epoch:2}  train loss {total_loss / len(X_train):.4f}  val loss {val_loss:.4f}  val macro-F1 {val_f1:.3f}")
    if val_f1 > best_f1:
        best_f1, best_state, best_epoch = val_f1, copy.deepcopy(model.state_dict()), epoch
print(f"best epoch: {best_epoch} (val macro-F1 {best_f1:.3f})")
model.load_state_dict(best_state)
# 4. Per-class report on validation
with torch.no_grad():
    predicted = model(X_val).argmax(dim=1)
print(classification_report(y_val, predicted, target_names=LABELS, digits=3))
print("confusion matrix (rows = true, columns = predicted):", LABELS)
print(confusion_matrix(y_val, predicted))

# 5. Save the model and the vectorizers
MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "line_classifier.pt")
joblib.dump((line_vec, prev_vec), MODEL_DIR / "vectorizers.joblib")