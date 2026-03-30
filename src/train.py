import os
import json
import numpy as np
from data import load_data
from model import NeuralNetwork, ACTIVATIONS
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# =========================
# 📁 RUN DIRECTORY CREATOR
# =========================

def get_next_run_dir(base="results"):
    os.makedirs(base, exist_ok=True)

    runs = [d for d in os.listdir(base) if d.startswith("run_")]
    ids = [int(r.split("_")[1]) for r in runs if r.split("_")[1].isdigit()]

    next_id = max(ids) + 1 if ids else 1
    run_dir = os.path.join(base, f"run_{next_id:03d}")

    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, "plots"))

    return run_dir

# =========================
# 🚀 TRAIN PIPELINE
# =========================

def train_pipeline(config):
    print("\n📥 Loading data...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()

    # =========================
    # 🔴 FIX: CREATE VALIDATION SET
    # =========================

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=config["seed"]
    )

    run_dir = get_next_run_dir()
    print(f"📁 Saving run to: {run_dir}")

    # =========================
    # 🧠 BUILD MODEL
    # =========================

    layers = []
    acts = []

    for l in config["layers"]:
        layers.append(l["neurons"])

        act_name = l["acts"]

        # ✅ SAFETY CHECK
        if act_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {act_name}")

        acts.append(ACTIVATIONS[act_name]["id"])

    model = NeuralNetwork(
        X_train, Y_train,
        layers=layers,
        activations=acts,
        lr=config["lr"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        seed=config["seed"]
    )

    # =========================
    # 🏋️ TRAIN (WITH VALIDATION)
    # =========================

    print("\n🚀 Training started...\n")

    model.train(
        X_val=X_val,
        Y_val=Y_val,
        patience=config.get("patience", 5)
    )

    # =========================
    # 📊 EVALUATE (ONLY TEST SET)
    # =========================

    y_true = np.argmax(Y_test, axis=1)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n📊 Test Accuracy: {accuracy:.4f}")
    print(f"📊 F1 Score: {f1:.4f}")

    # =========================
    # 💾 SAVE METRICS
    # =========================

    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "train_final_accuracy": float(model.train_acc[-1]),
        "lr": config["lr"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "layers": layers
    }

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # =========================
    # 💾 SAVE CONFIG
    # =========================

    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # =========================
    # 💾 SAVE MODEL
    # =========================

    model.save_model(f"{run_dir}/model.npz")

    # =========================
    # 📈 SAVE PLOTS
    # =========================

    model.plot_loss(save_path=f"{run_dir}/plots/loss.png")
    model.plot_accuracy(save_path=f"{run_dir}/plots/accuracy.png")
    model.plot_confusion(X_test, Y_test, save_path=f"{run_dir}/plots/confusion_matrix.png")
    model.plot_confidence(X_test, save_path=f"{run_dir}/plots/confidence.png")

    print("📊 Plots saved")

    # =========================
    # 🏁 DONE
    # =========================

    return model, X_test, Y_test, run_dir