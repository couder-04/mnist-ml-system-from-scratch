import argparse
import subprocess
import sys

# =========================
# 🧠 HELPER: SAFE RUN
# =========================

def run_command(cmd):
    try:
        print(f"\n⚡ Running: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
    except FileNotFoundError:
        print("❌ Required tool not found (python/streamlit)")

# =========================
# 🚀 TRAIN
# =========================

def run_train():
    print("🚀 Running Training...\n")

    cmd = [
        sys.executable, "src/main.py",
        "--mode", "single",
        "--layers", "128", "64", "10",
        "--acts", "relu", "relu", "linear"
    ]

    run_command(cmd)

# =========================
# 🔁 GRID SEARCH
# =========================

def run_grid():
    print("🔁 Running Grid Search...\n")

    cmd = [
        sys.executable, "src/main.py",
        "--mode", "grid"
    ]

    run_command(cmd)

# =========================
# 📊 COMPARE
# =========================

def run_compare():
    print("📊 Comparing Runs...\n")

    cmd = [
        sys.executable, "src/compare.py"
    ]

    run_command(cmd)

# =========================
# 🌐 APP
# =========================

def run_app():
    print("🌐 Launching App...\n")

    cmd = ["streamlit", "run", "src/app.py"]
    run_command(cmd)

# =========================
# 📊 DASHBOARD
# =========================

def run_dashboard():
    print("📊 Launching Dashboard...\n")

    cmd = ["streamlit", "run", "src/dashboard.py"]
    run_command(cmd)

# =========================
# 🎯 ARG PARSER
# =========================

def parse():
    parser = argparse.ArgumentParser(description="🔥 ML Project Runner")

    parser.add_argument(
        "command",
        choices=["train", "grid", "compare", "app", "dashboard"],
        help="What to run"
    )

    return parser.parse_args()

# =========================
# ▶️ MAIN
# =========================

def main():
    args = parse()

    if args.command == "train":
        run_train()

    elif args.command == "grid":
        run_grid()

    elif args.command == "compare":
        run_compare()

    elif args.command == "app":
        run_app()

    elif args.command == "dashboard":
        run_dashboard()

# =========================

if __name__ == "__main__":
    main()