import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 📊 LOAD ALL RUNS
# =========================

def load_runs(base="results"):
    if not os.path.exists(base):
        return pd.DataFrame()

    runs = [d for d in os.listdir(base) if d.startswith("run_")]
    runs = sorted(runs)

    data = []

    for r in runs:
        metrics_path = os.path.join(base, r, "metrics.json")

        if not os.path.exists(metrics_path):
            continue

        try:
            with open(metrics_path) as f:
                m = json.load(f)

            m["run"] = r
            data.append(m)

        except Exception:
            continue

    return pd.DataFrame(data)


# =========================
# 🏆 RANK MODELS
# =========================

def rank_models(df, metric="accuracy"):
    if df.empty:
        return df, None

    if metric not in df.columns:
        raise ValueError(f"❌ Metric '{metric}' not found in dataframe")

    df = df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    best = df.iloc[0]

    print("\n🏆 BEST MODEL:")
    print(best)

    return df, best


# =========================
# 📈 COMPARISON PLOTS
# =========================

def plot_comparison(df, save_dir="results/comparison"):
    if df.empty:
        print("⚠️ No data to plot")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Accuracy plot
    if "accuracy" in df.columns:
        plt.figure()
        plt.bar(df["run"], df["accuracy"])
        plt.title("Accuracy Comparison")
        plt.xlabel("Run")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "accuracy.png"))
        plt.close()

    # F1 plot
    if "f1" in df.columns:
        plt.figure()
        plt.bar(df["run"], df["f1"])
        plt.title("F1 Score Comparison")
        plt.xlabel("Run")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "f1.png"))
        plt.close()

    print("📊 Comparison plots saved")


# =========================
# 💾 SAVE SUMMARY
# =========================

def save_summary(df, best, save_dir="results/comparison"):
    if df.empty or best is None:
        print("⚠️ Nothing to save")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Save CSV
    df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    # Save best model info
    with open(os.path.join(save_dir, "best_model.json"), "w") as f:
        json.dump(best.to_dict(), f, indent=4)

    print("💾 Summary saved")


# =========================
# 🧠 INSIGHTS (BONUS 🔥)
# =========================

def print_insights(df):
    if df.empty:
        return

    print("\n🧠 INSIGHTS:")
    print(f"Total Runs: {len(df)}")

    if "accuracy" in df.columns:
        print(f"Best Accuracy: {df['accuracy'].max():.4f}")
        print(f"Average Accuracy: {df['accuracy'].mean():.4f}")

    if "f1" in df.columns:
        print(f"Best F1 Score: {df['f1'].max():.4f}")
        print(f"Average F1 Score: {df['f1'].mean():.4f}")


# =========================
# 🚀 MAIN COMPARE FUNCTION
# =========================

def compare_runs(base="results", metric="accuracy"):
    df = load_runs(base)

    if df.empty:
        print("⚠️ No runs found")
        return df

    df, best = rank_models(df, metric=metric)

    plot_comparison(df)
    save_summary(df, best)
    print_insights(df)

    return df