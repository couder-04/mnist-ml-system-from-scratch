import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

#  EVALUATE


def evaluate(model, X_test, Y_test, save_dir=None):
    y_true = np.argmax(Y_test, axis=1)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f" Accuracy: {accuracy:.4f}")
    print(f" F1 Score: {f1:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "evaluation.json"), "w") as f:
            import json
            json.dump({
                "accuracy": float(accuracy),
                "f1": float(f1)
            }, f, indent=4)

    return {"accuracy": accuracy, "f1": f1}



# MISCLASSIFIED SAMPLES


def show_misclassified(model, X_test, Y_test, save_dir=None, num=10):
    y_true = np.argmax(Y_test, axis=1)
    y_pred = model.predict(X_test)

    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print(" No misclassified samples!")
        return

    wrong_idx = wrong_idx[:num]

    plt.figure(figsize=(10, 4))

    for i, idx in enumerate(wrong_idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
        plt.axis("off")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "misclassified.png"))
        plt.close()
    else:
        plt.show()


#  CONFIDENCE HISTOGRAM

def plot_confidence(model, X_test, save_dir=None):
    logits = model.forward(X_test)[0]
    probs = model.softmax(logits)

    max_probs = np.max(probs, axis=1)

    plt.figure()
    plt.hist(max_probs, bins=20)
    plt.title("Prediction Confidence")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "confidence_eval.png"))
        plt.close()
    else:
        plt.show()

#  CONFUSION MATRIX 

def plot_confusion(model, X_test, Y_test, save_dir=None):
    y_true = np.argmax(Y_test, axis=1)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "confusion_eval.png"))
        plt.close()
    else:
        plt.show()