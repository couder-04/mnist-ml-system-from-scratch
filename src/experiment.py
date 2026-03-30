import itertools

# =========================
# ⚙️ DEFAULT SETTINGS
# =========================

DEFAULT_SEED = 42
NUM_CLASSES = 10

# =========================
# 🔁 GRID SEARCH CONFIG GENERATOR (CLEAN 🔥)
# =========================

def generate_configs(mode="normal"):
    """
    Returns a SMALL, meaningful set of configs
    mode: "fast" | "normal"
    """

    if mode == "fast":
        configs = [
            custom_config(lr=0.003, layers=[128], epochs=5),
            custom_config(lr=0.003, layers=[128, 64], epochs=5),
            custom_config(lr=0.001, layers=[128, 64], epochs=5),
        ]

        print(f"⚡ FAST MODE → {len(configs)} configs")
        return configs

    # =========================
    # 🎯 NORMAL MODE (BEST FOR GITHUB)
    # =========================

    configs = [
        # 1️⃣ baseline
        custom_config(lr=0.003, layers=[128], epochs=10),

        # 2️⃣ deeper
        custom_config(lr=0.003, layers=[128, 64], epochs=10),

        # 3️⃣ wider
        custom_config(lr=0.003, layers=[256, 128], epochs=10),

        # 4️⃣ lower lr
        custom_config(lr=0.001, layers=[128, 64], epochs=10),

        # 5️⃣ gelu test
        custom_config(lr=0.003, layers=[128, 64], acts=["gelu", "gelu"], epochs=10),

        # 6️⃣ regularization test
        custom_config(lr=0.003, layers=[128, 64], l2_lambda=0.001, epochs=10),

        # 7️⃣ small model (speed baseline)
        custom_config(lr=0.003, layers=[64], epochs=8),

        # 8️⃣ slightly deeper
        custom_config(lr=0.003, layers=[128, 64, 32], epochs=10),
    ]

    print(f"🔁 Generated {len(configs)} configs (clean mode)")

    return configs


# =========================
# 🎯 CUSTOM CONFIG BUILDER
# =========================

def custom_config(
    lr=0.003,
    layers=[128, 64],
    acts=None,
    epochs=10,
    bs=64,
    l2_lambda=0.0,
    patience=5,
    seed=42
):
    if acts is None:
        acts = ["relu"] * len(layers)

    if len(layers) != len(acts):
        raise ValueError("❌ Layers and activations must match")

    valid_acts = ["linear", "relu", "gelu", "elu", "leaky_relu"]

    config_layers = []

    for n, a in zip(layers, acts):
        a = a.lower()

        if a not in valid_acts:
            raise ValueError(f"❌ Invalid activation: {a}")

        config_layers.append({
            "neurons": n,
            "acts": a
        })

    # Output layer
    config_layers.append({"neurons": NUM_CLASSES, "acts": "linear"})

    return {
        "lr": lr,
        "epochs": epochs,
        "batch_size": bs,
        "seed": seed,
        "l2_lambda": l2_lambda,
        "patience": patience,
        "layers": config_layers
    }


# =========================
# 🧪 QUICK PRESETS
# =========================

def fast_debug_config():
    """Quick test config"""
    return custom_config(
        lr=0.003,
        layers=[64],
        acts=["relu"],
        epochs=5,
        bs=32
    )


def high_accuracy_config():
    """Stronger config"""
    return custom_config(
        lr=0.001,
        layers=[256, 128],
        acts=["relu", "relu"],
        epochs=20,
        bs=64,
        l2_lambda=0.001
    )