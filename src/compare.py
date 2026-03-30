import os
import json
import pandas as pd
import matplotlib.pyplot as plt

#  LOAD ALL RUNS


def load_runs(base="results"):
    if not os.path.exists(base):
        print(" Results folder not found")
        return pd.DataFrame()

    runs = sorted([
        d for d in os.listdir(base)
        if d.startswith("run_") and os.path.isdir(os.path.join(base, d))
    ])

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

        except Exception as e:
            print(f" Skipping {r}: {e}")

    df = pd.DataFrame(data)

    return df


#  RANK MODELS


def rank_models(df, metric="accuracy"):
    if df.empty:
        return df, None

    if metric not in df.columns:
        raise ValueError(f" Metric '{metric}' not found")

    df = df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    best = df.iloc[0]

    print("\n BEST MODEL:")
    print(best.to_string())

    return df, best



#  COMPARISON PLOTS

def plot_comparison(df, save_dir="results/comparison"):
    if df.empty:
        print(" No data to plot")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Accuracy
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

    # F1 Score
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

    print(" Plots saved → results/comparison/")


#  SAVE SUMMARY

def save_summary(df, best, save_dir="results/comparison"):
    if df.empty or best is None:
        print(" Nothing to save")
        return

    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    with open(os.path.join(save_dir, "best_model.json"), "w") as f:
        json.dump(best.to_dict(), f, indent=4)

    print(" Summary saved")


#  INSIGHTS

def print_insights(df):
    if df.empty:
        return

    print("\n INSIGHTS")
    print("-" * 40)

    print(f"Total Runs: {len(df)}")

    if "accuracy" in df.columns:
        print(f"Best Accuracy: {df['accuracy'].max():.4f}")
        print(f"Average Accuracy: {df['accuracy'].mean():.4f}")

    if "f1" in df.columns:
        print(f"Best F1 Score: {df['f1'].max():.4f}")
        print(f"Average F1 Score: {df['f1'].mean():.4f}")


#  MAIN FUNCTION


def compare_runs(base="results", metric="accuracy"):
    print("\n Comparing Runs...\n")

    df = load_runs(base)

    if df.empty:
        print(" No runs found")
        return df

    df, best = rank_models(df, metric)

    print("\n ALL RUNS:\n")
    print(df.to_string(index=False))

    plot_comparison(df)
    save_summary(df, best)
    print_insights(df)

    print("\n Comparison complete!")

    return df

#  RUN DIRECTLY

if __name__ == "__main__":
    compare_runs()