"""Shared helpers for the chord-quality models (SVM, MLP, distilled MLP)."""
import os

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skl2onnx import to_onnx
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

N_FEATURES = 81
SEED = 42


def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_kb(path):
    return os.path.getsize(path) / 1e3


# ---------- data ----------

def load_csv(csv_path, classes):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = np.array([classes.index(lbl) for lbl in df["label"]], dtype=np.int64)
    return X, y


def make_splits(X, y, test_fraction=0.2, val_fraction=0.15, seed=SEED):
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, random_state=seed, stratify=y_rest)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def print_summary(y, splits, classes):
    counts = np.bincount(y, minlength=len(classes))
    print(f"Samples: {len(y)} total | {len(splits['y_train'])} train | "
          f"{len(splits['y_val'])} val | {len(splits['y_test'])} test")
    print(f"Class counts: {dict(zip(classes, counts.tolist()))}\n")

    plt.figure(figsize=(8, 4))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Classes")
    plt.ylabel("Sample Count")
    plt.title("Class Distribution Summary")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def report_predictions(name, y_true, y_pred, classes):
    print(f"{name} accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred), "\n")


# ---------- SVM ----------

def train_svm(splits, C=1.0, gamma="scale", kernel="rbf"):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=SEED),
    )
    model.fit(splits["X_train"], splits["y_train"])
    return model


def svm_size(pipeline):
    svc = pipeline.named_steps["svc"]
    arrays = [svc.support_vectors_, svc.dual_coef_, svc.intercept_]
    return {
        "n_sv": svc.support_vectors_.shape[0],
        "n_numbers": sum(a.size for a in arrays),
        "n_bytes": sum(a.nbytes for a in arrays),
    }


def export_svm_onnx(pipeline, X_sample, path):
    onx = to_onnx(pipeline, X_sample[:1], target_opset=15, options={SVC: {"zipmap": False}})
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return file_kb(path)


# ---------- MLP ----------

class Standardize(nn.Module):
    """Fixed (x - mean) / scale, stored as buffers so it exports with the model."""
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.scale

def make_mlp(hidden, n_classes, splits, in_dim=N_FEATURES):
    sc = StandardScaler().fit(splits["X_train"])
    layers, d = [Standardize(sc.mean_, sc.scale_)], N_FEATURES
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def to_loader(X, y, batch_size=32, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def torch_predict(model, X):
    model.eval()
    return model(torch.from_numpy(X)).argmax(dim=1).numpy()


def train_mlp(model, loader, splits, epochs=50, lr=1e-3, log_every=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()
        if epoch % log_every == 0 or epoch == epochs:
            val_acc = accuracy_score(splits["y_val"], torch_predict(model, splits["X_val"]))
            print(f"epoch {epoch:>3} | val acc {val_acc:.2%}")
    return model


def export_torch_onnx(model, path):
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, N_FEATURES), path,
        input_names=["input"], output_names=["output"], opset_version=15,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return file_kb(path)


# ---------- distillation ----------

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T ** 2)
    return alpha * hard + (1 - alpha) * soft


def distill_mlp(student, teacher, loader, epochs=50, lr=1e-3, T=4.0, alpha=0.5):
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(epochs):
        student.train()
        for bx, by in loader:
            with torch.no_grad():
                t_logits = teacher(bx)
            opt.zero_grad()
            distillation_loss(student(bx), t_logits, by, T, alpha).backward()
            opt.step()
    return student


def sweep_distillation(teacher, splits, candidates, n_classes, tolerance=0.02,
                       epochs=50, T=4.0, alpha=0.5):
    loader = to_loader(splits["X_train"], splits["y_train"], shuffle=True)
    teacher_val = accuracy_score(splits["y_val"], torch_predict(teacher, splits["X_val"]))
    threshold = teacher_val - tolerance
    print(f"Teacher: {count_params(teacher):,} params, val acc {teacher_val:.2%}")
    print(f"Threshold: {threshold:.2%}\n")
    print(f"{'Hidden':>12} | {'Params':>7} | {'Val Acc':>8} | OK")
    print("-" * 42)

    results = []
    for hidden in candidates:
        torch.manual_seed(SEED)
        student = distill_mlp(make_mlp(hidden, n_classes, splits), teacher, loader,
                              epochs=epochs, T=T, alpha=alpha)
        val_acc = accuracy_score(splits["y_val"], torch_predict(student, splits["X_val"]))
        n = count_params(student)
        ok = val_acc >= threshold
        results.append({"hidden": hidden, "params": n, "val_acc": val_acc,
                        "ok": ok, "model": student})
        print(f"{str(hidden):>12} | {n:>7,} | {val_acc:>7.2%} | {'yes' if ok else 'no'}")

    accepted = [r for r in results if r["ok"]]
    return min(accepted, key=lambda r: r["params"]) if accepted else None


# ---------- ONNX check ----------

def onnx_check(path, X, y, classes, is_svm=False, n_samples=10, seed=0):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    idx = np.random.default_rng(seed).choice(len(X), size=min(n_samples, len(X)), replace=False)
    outs = sess.run(None, {name: X[idx]})

    if is_svm:
        probs = outs[1]  # zipmap disabled: [labels, probabilities]
    else:
        logits = outs[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)

    print(f"{'#':>3} | {'True':>6} | {'Predicted':>9} | {'Conf':>6} | Correct")
    print("-" * 46)
    for i, (row, p) in enumerate(zip(idx, probs)):
        pred, true = classes[int(p.argmax())], classes[int(y[row])]
        print(f"{i:>3} | {true:>6} | {pred:>9} | {p.max():>5.1%} | {'yes' if pred == true else 'no'}")
    acc = np.mean([classes[int(p.argmax())] == classes[int(y[r])] for r, p in zip(idx, probs)])
    print(f"\nAccuracy on {len(idx)} samples: {acc:.2%}")

def get_probs_torch(model, X):
    """Softmax probabilities from a PyTorch model."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()


def expected_calibration_error(probs, y_true, n_bins=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def confidence_report(name, probs, y_true, classes):
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == y_true

    print(f"== {name} ==")
    print(f"{'Class':>6} | {'N':>4} | {'Acc':>7} | {'Mean conf':>9} | "
          f"{'Min conf (ok)':>13} | {'Conf (wrong)':>12} | {'P(true)':>7}")
    print("-" * 74)
    for k, cls in enumerate(classes):
        m = y_true == k
        if m.sum() == 0:
            continue
        ok = m & correct
        wrong = m & ~correct
        min_ok = conf[ok].min() if ok.any() else float("nan")
        conf_wrong = conf[wrong].mean() if wrong.any() else float("nan")
        p_true = probs[m, k].mean()
        print(f"{cls:>6} | {m.sum():>4} | {correct[m].mean():>6.2%} | "
              f"{conf[m].mean():>8.2%} | {min_ok:>12.2%} | {conf_wrong:>11.2%} | {p_true:>6.3f}")

    print(f"\nOverall: accuracy {correct.mean():.2%}, mean conf {conf.mean():.2%}, "
          f"ECE {expected_calibration_error(probs, y_true):.4f}")
    low = np.where((conf < 0.6) | ~correct)[0]
    print(f"Low-confidence or wrong samples: {len(low)}\n")
