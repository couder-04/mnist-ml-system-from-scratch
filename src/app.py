import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

from model import NeuralNetwork
from data import load_data
from utils import get_best_run

st.set_page_config(page_title="MNIST Canvas", layout="centered")

st.title(" MNIST Digit Recognizer")
st.markdown("Draw a digit (0–9) and let the model predict!")

#  LOAD BEST MODEL

@st.cache_resource
def load_model():
    best_run, acc = get_best_run()

    if best_run is None:
        st.error(" No trained model found. Run training first.")
        return None

    model_path = os.path.join("results", best_run, "model.npz")

    # Dummy data (needed for model init)
    X_train, Y_train, _, _, _, _ = load_data()

    model = NeuralNetwork.load_model(model_path, X_train[:1], Y_train[:1])

    return model, best_run, acc


model_data = load_model()

if model_data is None:
    st.stop()

model, best_run, best_acc = model_data

st.success(f"Using Best Model → {best_run} (Accuracy: {best_acc:.4f})")

#  CANVAS

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

#  PREPROCESS

def preprocess(img):
    img = Image.fromarray(img.astype("uint8")).convert("L")

    #  convert to numpy
    img = np.array(img)

    #  threshold (binarize)
    img = (img > 50).astype(np.uint8) * 255

    #  crop to bounding box
    coords = np.column_stack(np.where(img > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img = img[y_min:y_max+1, x_min:x_max+1]

    #  resize to 20x20 (like MNIST)
    img = Image.fromarray(img).resize((20, 20))

    #  pad to 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = img

    img = padded.astype(np.float32)

    #  normalize
    img /= 255.0

    return img.reshape(1, -1), img


#  PREDICTION

if canvas.image_data is not None:

    x, img = preprocess(canvas.image_data)

    logits = model.forward(x)[0]
    probs = model.softmax(logits)[0]

    pred = np.argmax(probs)

    st.subheader(f"🎯 Prediction: {pred}")

    #  PROBABILITY DISPLAY

    st.markdown("### 📊 Confidence Scores")

    prob_df = {
        "Digit": list(range(10)),
        "Probability": probs
    }

    st.bar_chart(prob_df, x="Digit", y="Probability")

    #  PROCESSED IMAGE
    

    st.markdown("### 🖼️ Processed Input")
    st.image(img, width=150)


#  CLEAR BUTTON


if st.button("🧹 Clear Canvas"):
    st.rerun()