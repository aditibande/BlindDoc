import os
os.environ["MPLBACKEND"] = "Agg"
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.tree import DecisionTreeClassifier as SkTree
from sklearn.metrics import accuracy_score
from concrete.ml.sklearn import LogisticRegression as FheLogReg
from concrete.ml.sklearn import DecisionTreeClassifier as FheTree

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

configs = [
    ("LogReg", "n_bits=4", FheLogReg, {"n_bits": 4}, SkLogReg(max_iter=1000), True, len(X_test)),
    ("LogReg", "n_bits=6", FheLogReg, {"n_bits": 6}, SkLogReg(max_iter=1000), True, len(X_test)),
    ("LogReg", "n_bits=8", FheLogReg, {"n_bits": 8}, SkLogReg(max_iter=1000), True, len(X_test)),
    ("DecisionTree", "depth=3", FheTree, {"max_depth": 3, "n_bits": 6}, SkTree(max_depth=3, random_state=0), False, 20),
    ("DecisionTree", "depth=4", FheTree, {"max_depth": 4, "n_bits": 6}, SkTree(max_depth=4, random_state=0), False, 20),
]

rows = []
for name, label, cls, kwargs, sk_model, scaled, n_eval in configs:
    Xtr = X_train_s if scaled else X_train
    Xte = X_test_s if scaled else X_test

    sk_model.fit(Xtr, y_train)
    plain_acc = accuracy_score(y_test, sk_model.predict(Xte))

    t = time.time()
    sk_model.predict(Xte)
    plain_latency = (time.time() - t) / len(Xte)

    model = cls(**kwargs)
    model.fit(Xtr, y_train)

    t = time.time()
    model.compile(Xtr)
    compile_time = time.time() - t

    t = time.time()
    model.fhe_circuit.keygen()
    keygen_time = time.time() - t

    clear_pred = model.predict(Xte)
    clear_acc = accuracy_score(y_test, clear_pred)

    t = time.time()
    fhe_pred = model.predict(Xte[:n_eval], fhe="execute")
    fhe_latency = (time.time() - t) / n_eval

    stats = model.fhe_circuit.statistics
    rows.append({
        "model": name,
        "config": label,
        "plain_acc": round(plain_acc, 4),
        "quantized_clear_acc": round(clear_acc, 4),
        "fhe_eval_samples": n_eval,
        "fhe_acc_on_subset": round(accuracy_score(y_test[:n_eval], fhe_pred), 4),
        "agreement_with_clear": round(float(np.mean(fhe_pred == clear_pred[:n_eval])), 4),
        "plain_ms_per_sample": round(plain_latency * 1000, 5),
        "fhe_ms_per_sample": round(fhe_latency * 1000, 2),
        "compile_s": round(compile_time, 2),
        "keygen_s": round(keygen_time, 2),
        "bootstraps": stats["programmable_bootstrap_count"],
        "bootstrap_key_mb": round(stats["size_of_bootstrap_keys"] / 1e6, 1),
        "keyswitch_key_mb": round(stats["size_of_keyswitch_keys"] / 1e6, 1),
    })
    print(rows[-1], flush=True)

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/results.csv", index=False)

labels = df["model"] + "\n" + df["config"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].bar(labels, df["fhe_ms_per_sample"], color="tab:red")
axes[0].set_yscale("log")
axes[0].set_ylabel("ms per prediction (log scale)")
axes[0].set_title("Encrypted inference latency")

axes[1].bar(labels, df["bootstrap_key_mb"], color="tab:purple")
axes[1].set_ylabel("MB")
axes[1].set_title("Bootstrap key size")

x = np.arange(len(df))
axes[2].bar(x - 0.2, df["plain_acc"], width=0.4, label="plaintext")
axes[2].bar(x + 0.2, df["quantized_clear_acc"], width=0.4, label="quantized")
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels)
axes[2].set_ylim(0.85, 1.0)
axes[2].set_ylabel("accuracy (y-axis starts at 0.85)")
axes[2].set_title("Accuracy (test set, n=114)")
axes[2].legend()

for ax in axes:
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig("results/results.png", dpi=150)
print(df.to_string())
