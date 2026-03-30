import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="MNIST Dashboard", layout="wide")

BASE_DIR = os.path.abspath("results")

st.title(" MNIST Experiment Dashboard")


#  LOAD RUN DATA


def load_runs():
    if not os.path.exists(BASE_DIR):
        return pd.DataFrame()

    runs = sorted([
        d for d in os.listdir(BASE_DIR)
        if d.startswith("run_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    data = []

    for r in runs:
        metrics_path = os.path.join(BASE_DIR, r, "metrics.json")
        config_path  = os.path.join(BASE_DIR, r, "config.json")

        if not os.path.exists(metrics_path):
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)

            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {}

            metrics["run"] = r
            metrics["layers"] = str(metrics.get("layers", []))
            metrics["lr"] = metrics.get("lr", None)
            metrics["epochs"] = metrics.get("epochs", None)
            metrics["batch_size"] = metrics.get("batch_size", None)

            data.append(metrics)

        except Exception:
            continue

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.sort_values(by="accuracy", ascending=False)

    return df


df = load_runs()

#  EMPTY CHECK


if df.empty:
    st.warning(" No runs found. Train a model first.")
    st.stop()

#  SIDEBAR CONTROLS


st.sidebar.header(" Controls")

min_acc = st.sidebar.slider("Minimum Accuracy", 0.0, 1.0, 0.0)
show_top_k = st.sidebar.slider("Top K Runs", 1, len(df), min(5, len(df)))

filtered_df = df[df["accuracy"] >= min_acc].head(show_top_k)

#  TABLE VIEW


st.subheader(" Experiments Overview")
st.dataframe(filtered_df, use_container_width=True)


#  BEST MODEL

best = df.iloc[0]

st.subheader("🏆 Best Model")
st.success(
    f"Run: {best['run']} | Accuracy: {best['accuracy']:.4f} | F1: {best.get('f1', 0):.4f}"
)


#  COMPARISON PLOTS


col1, col2 = st.columns(2)

with col1:
    st.subheader(" Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(filtered_df["run"], filtered_df["accuracy"])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Run")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader(" F1 Score Comparison")
    fig, ax = plt.subplots()
    ax.bar(filtered_df["run"], filtered_df.get("f1", 0))
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Run")
    plt.xticks(rotation=45)
    st.pyplot(fig)


#  TREND PLOT


st.subheader(" Accuracy Trend")

fig, ax = plt.subplots()
ax.plot(df["run"], df["accuracy"], marker="o")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Run")
plt.xticks(rotation=45)
st.pyplot(fig)


# RUN DETAILS


st.subheader(" Inspect Run")

selected_run = st.selectbox("Select run", df["run"].tolist())

run_path = os.path.join(BASE_DIR, selected_run)

st.caption(f" Loading from: {run_path}")

#  CONFIG VIEW

config_path = os.path.join(run_path, "config.json")

if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)

    st.markdown("###  Configuration")
    st.json(config)
else:
    st.info("No config found for this run")

#  METRICS VIEW


metrics_path = os.path.join(run_path, "metrics.json")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.markdown("###  Metrics")
    st.json(metrics)


#  SHOW PLOTS


st.subheader(" Run Plots")

plots_dir = os.path.join(run_path, "plots")

plot_files = [
    ("loss.png", "Loss Curve"),
    ("accuracy.png", "Accuracy Curve"),
    ("confusion_matrix.png", "Confusion Matrix"),
    ("confidence.png", "Confidence Histogram")
]

cols = st.columns(2)

for i, (pf, title) in enumerate(plot_files):
    path = os.path.join(plots_dir, pf)

    with cols[i % 2]:
        if os.path.exists(path):
            try:
                st.image(Image.open(path), caption=title)
            except Exception as e:
                st.error(f"Error loading {pf}: {e}")
        else:
            st.warning(f"{pf} not found at {path}")


#  INSIGHTS


st.subheader(" Insights")

st.write(f"Total Runs: {len(df)}")
st.write(f"Best Accuracy: {df['accuracy'].max():.4f}")
st.write(f"Average Accuracy: {df['accuracy'].mean():.4f}")