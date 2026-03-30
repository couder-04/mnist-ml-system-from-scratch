import argparse
import numpy as np
from PIL import Image
from model import NeuralNetwork
from utils import get_best_run
import os
import matplotlib.pyplot as plt


#  PREPROCESS


def preprocess(img_path, show=False):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))

    img = np.array(img).astype(np.float32)

    # invert if background is white
    if np.mean(img) > 127:
        img = 255 - img

    img /= 255.0

    if show:
        plt.imshow(img, cmap="gray")
        plt.title("Processed Image")
        plt.axis("off")
        plt.show()

    return img.reshape(1, -1), img



#  LOAD BEST MODEL


def load_best_model():
    best_run, _ = get_best_run()

    if best_run is None:
        raise ValueError(" No trained model found")

    model_path = os.path.join("results", best_run, "model.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    
    X_dummy = np.zeros((1, 784))
    Y_dummy = np.zeros((1, 10))

    model = NeuralNetwork.load_model(model_path, X_dummy, Y_dummy)

    print(f" Loaded model from: {model_path}")

    return model



#  PREDICT


def predict_image(model, img_path, show=False):
    x, img = preprocess(img_path, show=show)

    logits = model.forward(x)[0]
    probs = model.softmax(logits)[0]

    pred = np.argmax(probs)
    confidence = np.max(probs)

    return pred, confidence, probs


# MAIN


def main():
    parser = argparse.ArgumentParser(description="Predict digit from image")

    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--show", action="store_true", help="Show processed image")
    parser.add_argument("--topk", type=int, default=10, help="Top-k probabilities to display")

    args = parser.parse_args()

    model = load_best_model()

    pred, confidence, probs = predict_image(model, args.image, show=args.show)

    print(f"\n Prediction: {pred}")
    print(f" Confidence: {confidence:.4f}")

    print("\n Top Probabilities:")

    topk_idx = np.argsort(probs)[::-1][:args.topk]

    for i in topk_idx:
        print(f"{i}: {probs[i]:.4f}")



#  ENTRY


if __name__ == "__main__":
    main()