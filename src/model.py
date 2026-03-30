import numpy as np
from optimizers import get_optimizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# =========================
# ⚡ ACTIVATIONS
# =========================

ACTIVATIONS = {
    "linear": {"id": 0, "alpha": None},
    "relu": {"id": 1, "alpha": None},
    "elu": {"id": 2, "alpha": 0.001},
    "gelu": {"id": 3, "alpha": None},
    "leaky_relu": {"id": 4, "alpha": 0.001}
}

ID_TO_ACT = {v["id"]: k for k, v in ACTIVATIONS.items()}

ACT_FUNCS = {
    0: lambda x, a=None: x,
    1: lambda x, a=None: np.maximum(0, x),
    2: lambda x, a=0.001: np.where(x > 0, x, a * (np.exp(x) - 1)),
    3: lambda x, a=None: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))),
    4: lambda x, a=0.001: np.where(x > 0, x, a * x)
}

ACT_DERIVS = {
    0: lambda x, a=None: np.ones_like(x),
    1: lambda x, a=None: (x > 0).astype(float),
    2: lambda x, a=0.001: np.where(x > 0, 1, a*np.exp(x)),
    3: lambda x, a=None: (
        0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))) +
        0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))**2)
        * np.sqrt(2/np.pi) * (1 + 3*0.044715*x**2)
    ),
    4: lambda x, a=0.001: np.where(x > 0, 1, a)
}

# =========================
# 🔁 BATCHING
# =========================

def create_batches(X, Y, bs):
    idx = np.random.permutation(X.shape[0])
    for i in range(0, len(idx), bs):
        batch = idx[i:i+bs]
        yield X[batch], Y[batch]

# =========================
# 🧠 MODEL
# =========================

class NeuralNetwork:
    def __init__(self, X, Y, layers, activations,
                 lr=0.003, epochs=20, batch_size=64, seed=42, l2_lambda=0.001):

        np.random.seed(seed)

        self.X, self.Y = X, Y
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.initial_lr = lr
        self.l2_lambda = l2_lambda

        self.layers = [X.shape[1]] + layers
        self.activations = activations

        self.W, self.B = [], []

        for i in range(len(self.layers) - 1):
            self.W.append(
                np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2/self.layers[i])
            )
            self.B.append(np.zeros((1, self.layers[i+1])))

        self.losses = []
        self.train_acc = []
        self.optimizer = get_optimizer("adam")  # 🔥 best default

    # =========================
    # 🔁 FORWARD
    # =========================

    def forward(self, X):
        activations, pre_activations = [X], []
        act = X

        for i in range(len(self.W) - 1):
            z = act @ self.W[i] + self.B[i]
            alpha = ACTIVATIONS[ID_TO_ACT[self.activations[i]]]["alpha"]

            act = ACT_FUNCS[self.activations[i]](z, alpha)

            pre_activations.append(z)
            activations.append(act)

        logits = act @ self.W[-1] + self.B[-1]
        activations.append(logits)

        return logits, activations, pre_activations

    # =========================
    # 🔥 SOFTMAX + LOSS
    # =========================

    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)

    def loss(self, probs, Y):
        probs = np.clip(probs, 1e-15, 1)
        data_loss = -np.mean(np.sum(Y * np.log(probs), axis=1))

        l2_loss = sum(np.sum(W**2) for W in self.W)

        return data_loss + self.l2_lambda * l2_loss

    # =========================
    # 🔙 BACKPROP
    # =========================

    def backward(self, X, Y):
        logits, activations, pre_activations = self.forward(X)
        probs = self.softmax(logits)

        n = X.shape[0]
        dZ = probs - Y

        for i in reversed(range(len(self.W))):
            dW = (activations[i].T @ dZ) / n
            dB = np.sum(dZ, axis=0, keepdims=True) / n

            # L2
            dW += self.l2_lambda * self.W[i]

            # Gradient clipping
            dW = np.clip(dW, -1, 1)
            dB = np.clip(dB, -1, 1)
            # Update weights
            self.W[i] = self.optimizer.step(self.W[i], dW, self.lr, idx=i)

            # Update biases
            self.B[i] = self.optimizer.step(self.B[i], dB, self.lr, idx=i + 1000)

            # 🔥 FORCE SHAPE (VERY IMPORTANT)
            if self.B[i].ndim == 1:
                self.B[i] = self.B[i].reshape(1, -1)       

            if i > 0:
                alpha = ACTIVATIONS[ID_TO_ACT[self.activations[i-1]]]["alpha"]
                dZ = (dZ @ self.W[i].T) * ACT_DERIVS[self.activations[i-1]](
                    pre_activations[i-1], alpha
                )

        return self.loss(probs, Y)

    # =========================
    # 🔮 PREDICT
    # =========================

    def predict(self, X):
        return np.argmax(self.softmax(self.forward(X)[0]), axis=1)

    def accuracy(self, X, Y):
        return np.mean(self.predict(X) == np.argmax(Y, axis=1))

    # =========================
    # 🚀 TRAIN
    # =========================

    def train(self, X_val=None, Y_val=None, patience=5):
        best_loss = float("inf")
        patience_counter = 0

        best_W, best_B = None, None  # ✅ store best model

        for epoch in range(self.epochs):

            self.lr = self.initial_lr * (0.95 ** epoch)

            perm = np.random.permutation(len(self.X))
            self.X = self.X[perm]
            self.Y = self.Y[perm]

            total_loss = 0
            batches = 0

            for xb, yb in create_batches(self.X, self.Y, self.batch_size):
                total_loss += self.backward(xb, yb)
                batches += 1

            total_loss /= max(batches, 1)

            acc = self.accuracy(self.X, self.Y)

            self.losses.append(total_loss)
            self.train_acc.append(acc)

            log = f"Epoch {epoch+1}: Loss={total_loss:.4f}, Acc={acc:.4f}"

            if X_val is not None:
                val_probs = self.softmax(self.forward(X_val)[0])
                val_loss = self.loss(val_probs, Y_val)
                val_acc = self.accuracy(X_val, Y_val)

                log += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0

                    # ✅ save best weights
                    best_W = [w.copy() for w in self.W]
                    best_B = [b.copy() for b in self.B]
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("⏹ Early stopping triggered")
                    break

            print(log)

        # ✅ restore best weights
        if best_W is not None:
            self.W = best_W
            self.B = best_B

    # =========================
    # 📈 PLOTS
    # =========================

    def plot_loss(self, save_path=None):
        plt.figure()
        plt.plot(self.losses)
        plt.title("Loss vs Epoch")
        plt.grid()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_accuracy(self, save_path=None):
        plt.figure()
        plt.plot(self.train_acc)
        plt.title("Accuracy vs Epoch")
        plt.grid()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_confusion(self, X, Y, save_path=None):
        cm = confusion_matrix(np.argmax(Y,1), self.predict(X))

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_confidence(self, X, save_path=None):
        probs = self.softmax(self.forward(X)[0])

        plt.figure()
        plt.hist(np.max(probs, axis=1), bins=20)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    # =========================
    # 💾 SAVE / LOAD
    # =========================
    def save_model(self, path="models/model.npz"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 🔥 Create true object arrays manually
        W_obj = np.empty(len(self.W), dtype=object)
        B_obj = np.empty(len(self.B), dtype=object)

        for i, w in enumerate(self.W):
            W_obj[i] = w

        for i, b in enumerate(self.B):
            # 🔥 FORCE SHAPE FIX
            if b.ndim == 1:
                b = b.reshape(1, -1)
            B_obj[i] = b

        np.savez(
            path,
            W=W_obj,
            B=B_obj,
            layers=self.layers,
            activations=self.activations
        )

        print(f"✅ Model saved → {path}")


    @staticmethod
    def load_model(path, X_dummy, Y_dummy):
        data = np.load(path, allow_pickle=True)

        model = NeuralNetwork(
            X_dummy,
            Y_dummy,
            layers=list(data["layers"][1:]),
            activations=list(data["activations"])
        )

        model.W = [w for w in data["W"]]
        model.B = [b for b in data["B"]]

        print("✅ Model loaded")

        return model