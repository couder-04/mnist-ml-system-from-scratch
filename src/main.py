import argparse
from train import train_pipeline
from evaluate import evaluate, show_misclassified, plot_confidence
from experiment import generate_configs
from compare import compare_runs, rank_models
import os

# =========================
# ⚠️ VALID ACTIVATIONS
# =========================

VALID = ["linear", "relu", "gelu", "elu", "leaky_relu"]

# =========================
# 🎯 ARG PARSER
# =========================

def parse():
    p = argparse.ArgumentParser(description="MNIST Neural Network Runner")

    # Modes
    p.add_argument("--mode", type=str, default="single",
                   choices=["single", "grid"],
                   help="Run mode: single or grid search")

    # Single run params
    p.add_argument("--layers", type=int, nargs="+",
                   help="Hidden layer sizes (e.g. 128 64)")
    p.add_argument("--acts", type=str, nargs="+",
                   help="Activation functions per layer")

    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

# =========================
# ⚙️ CONFIG BUILDER
# =========================

def build_config(a):
    if a.layers is None or a.acts is None:
        raise ValueError("❌ --layers and --acts are required for single mode")

    if len(a.layers) != len(a.acts):
        raise ValueError("❌ Layers and activations must match")

    layers_config = []

    for n, act in zip(a.layers, a.acts):
        act = act.lower()

        if act not in VALID:
            raise ValueError(f"❌ Invalid activation: {act}")

        layers_config.append({
            "neurons": n,
            "acts": act
        })

    return {
        "layers": layers_config,
        "lr": a.lr,
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "seed": a.seed,
        "patience": 5  # ✅ early stopping control
    }

# =========================
# 🚀 SINGLE RUN
# =========================

def run_single(args):
    print("\n🚀 Running Single Experiment...\n")

    config = build_config(args)

    model, X_test, Y_test, run_dir = train_pipeline(config)

    print("\n📊 Evaluating...\n")

    metrics = evaluate(model, X_test, Y_test, save_dir=run_dir)

    show_misclassified(model, X_test, Y_test, save_dir=run_dir)
    plot_confidence(model, X_test, save_dir=run_dir)

    print(f"\n✅ Done! Results saved in: {run_dir}")

# =========================
# 🔁 GRID SEARCH
# =========================

def run_grid():
    print("\n🔁 Running Grid Search...\n")

    configs = generate_configs()

    if not configs:
        print("❌ No configs generated")
        return

    for i, config in enumerate(configs):
        print(f"\n🚀 Experiment {i+1}/{len(configs)}")

        try:
            train_pipeline(config)
        except Exception as e:
            print(f"⚠️ Failed config {i+1}: {e}")
            continue

    print("\n📊 Comparing runs...\n")

    df = compare_runs()

    if df is None or df.empty:
        print("❌ No runs to compare")
        return

    ranked_df, best = rank_models(df)

    print("\n🏆 BEST MODEL:")
    print(best)

# =========================
# ▶️ MAIN
# =========================

if __name__ == "__main__":
    args = parse()

    if args.mode == "single":
        run_single(args)

    elif args.mode == "grid":
        run_grid()