import os
import json
import datetime

# =========================
# 📁 RUN DIRECTORY
# =========================

def get_next_run_dir(base="results"):
    os.makedirs(base, exist_ok=True)

    runs = [d for d in os.listdir(base) if d.startswith("run_")]

    ids = []
    for r in runs:
        try:
            ids.append(int(r.split("_")[1]))
        except:
            continue

    next_id = max(ids) + 1 if ids else 1
    run_dir = os.path.join(base, f"run_{next_id:03d}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    return run_dir


# =========================
# 💾 SAVE JSON
# =========================

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# =========================
# 📥 LOAD JSON
# =========================

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


# =========================
# 📝 SIMPLE LOGGER
# =========================

def get_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"

        print(full_msg)

        try:
            with open(log_path, "a") as f:
                f.write(full_msg + "\n")
        except Exception as e:
            print(f"⚠️ Logging failed: {e}")

    return log


# =========================
# 🏆 FIND BEST RUN
# =========================

def get_best_run(base="results"):
    if not os.path.exists(base):
        return None, None

    best_acc = -1
    best_run = None

    for r in sorted(os.listdir(base)):
        if not r.startswith("run_"):
            continue

        metrics_path = os.path.join(base, r, "metrics.json")

        if not os.path.exists(metrics_path):
            continue

        try:
            with open(metrics_path) as f:
                m = json.load(f)

            acc = m.get("accuracy", -1)

            if acc > best_acc:
                best_acc = acc
                best_run = r

        except Exception:
            continue

    return best_run, best_acc


# =========================
# 📊 LOAD ALL RUNS
# =========================

def load_all_runs(base="results"):
    if not os.path.exists(base):
        return []

    runs_data = []

    for r in sorted(os.listdir(base)):
        if not r.startswith("run_"):
            continue

        metrics_path = os.path.join(base, r, "metrics.json")

        if not os.path.exists(metrics_path):
            continue

        try:
            with open(metrics_path) as f:
                m = json.load(f)

            m["run"] = r
            runs_data.append(m)

        except Exception:
            continue

    # ✅ sort by accuracy descending
    runs_data.sort(key=lambda x: x.get("accuracy", 0), reverse=True)

    return runs_data