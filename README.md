#  MNIST Digit Classifier — Built from Scratch

> **A sophisticated, educational implementation of a fully-connected neural network using pure NumPy.** No PyTorch, no TensorFlow—just math, code, and deep learning fundamentals.

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

##  Vision

This project demonstrates that understanding neural networks *deeply* requires implementing them from first principles. Every component—forward propagation, backpropagation, gradient computation, optimization—is hand-coded and thoroughly documented.

**Perfect for:**
-  Learning neural network fundamentals
-  Understanding backpropagation in detail
-  Computer science coursework or interviews
-  Building intuition before using production frameworks

---

## ✨ Key Features

### 🧠 **Core Neural Network Implementation**
- ✅ **Custom Forward Propagation** — matrix operations, non-linearities, softmax output
- ✅ **Custom Backpropagation** — full gradient computation through all layers
- ✅ **Gradient Clipping & Stability** — prevents exploding gradients
- ✅ **L2 Regularization** — prevents overfitting
- ✅ **Learning Rate Decay** — adaptive learning schedules

### ⚡ **Flexible Architecture**
- **5 Activation Functions**: ReLU, Leaky ReLU, ELU, GELU, Linear
- **Configurable Layers**: Arbitrary hidden layer sizes and depths
- **Adam Optimizer** — Per-parameter adaptive learning rates (with momentum & RMSprop)
- **Batch Processing** — Efficient mini-batch gradient updates

### 📊 **Comprehensive Evaluation**
- Accuracy tracking during training
- Confusion matrices with per-class breakdown
- Misclassified sample visualization
- Prediction confidence histograms
- Loss & accuracy curves with convergence analysis

### 🎨 **Interactive Demo**
- **Streamlit Canvas App** — Draw digits in real-time and get predictions
- Confidence scores for each digit class
- Processed input visualization
- Automatic best-model selection

### 🔁 **Experimentation Framework**
- **Grid Search** — Hyperparameter exploration with automated runs
- **Run Comparison** — Rank models by accuracy, loss, and training efficiency
- **Dashboard** — Visual analysis of all experiments
- **Result Persistence** — Save & load trained models with full metadata

### 🛠️ **Production-Grade Code**
- Modular architecture with clear separation of concerns
- Comprehensive docstrings and inline comments
- Type hints and error handling
- Reproducibility with seed control
- Command-line interface for all operations

---

## 📊 Project Architecture

```
mnist-project/
│
├── src/                          # Core implementation
│   ├── model.py                 # Neural network class (forward, backward, optimization)
│   ├── train.py                 # Training pipeline with validation
│   ├── evaluate.py              # Evaluation metrics & visualizations
│   ├── data.py                  # MNIST loading & preprocessing
│   ├── optimizers.py            # Adam optimizer implementation
│   ├── main.py                  # CLI for single runs & grid search
│   ├── experiment.py            # Hyperparameter config generation
│   ├── compare.py               # Run comparison & ranking
│   ├── app.py                   # Streamlit interactive demo
│   ├── dashboard.py             # Results visualization dashboard
│   └── predict.py               # Batch inference
│
├── data/                         # MNIST dataset
│   ├── mnist_train.csv          # 60K training samples
│   └── mnist_test.csv           # 10K test samples
│
├── results/                      # Experiment outputs
│   ├── run_001/                 # Model, metrics, visualizations
│   ├── run_002/
│   └── ...
│
├── models/                       # Trained model checkpoints
│   └── model.npz
│
├── notebooks/                    # Jupyter analysis
│   └── EXPERIMENT.ipynb
│
├── run.py                        # High-level command runner
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1️⃣ **Installation**

```bash
git clone https://github.com/yourusername/mnist-scratch.git
cd mnist-scratch

pip install -r requirements.txt
```

### 2️⃣ **Train a Model**

```bash
# Single training run with custom architecture
python run.py train

# Or directly with arguments
python src/main.py --mode single \
  --layers 128 64 10 \
  --acts relu relu linear \
  --lr 0.003 \
  --epochs 50 \
  --batch_size 64
```

### 3️⃣ **Run Hyperparameter Grid Search**

```bash
python run.py grid
```

Automatically explores multiple configurations and ranks results.

### 4️⃣ **Launch Interactive Demo**

```bash
python run.py app
```

Draw a digit in your browser and get instant predictions!

### 5️⃣ **View Experiment Dashboard**

```bash
python run.py dashboard
```

Compare all runs side-by-side with metrics and visualizations.

---

## 🔬 Technical Deep Dive

### Forward Propagation

```python
# Input → Hidden Layers → Output Probabilities
for layer in range(num_layers):
    z = activation(x @ W + b)      # Linear combination + activation
    x = z                           # Pass to next layer
probs = softmax(final_z)            # Output probabilities
```

### Backpropagation

```python
# Gradient computation flowing backward through network
dL/dW = (dL/dZ @ A.T) / batch_size  # Weight gradient
dL/dB = sum(dL/dZ) / batch_size      # Bias gradient
dL/dA = dL/dZ @ W.T                  # Propagate to previous layer
```

### Adam Optimizer

Combines momentum and adaptive learning rates for faster convergence:

```
m_t = β₁ * m_(t-1) + (1 - β₁) * g_t      # Momentum
v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²     # Adaptive rate
θ_t = θ_(t-1) - α * m_t / (√v_t + ε)    # Parameter update
```

### Loss Function

Cross-entropy with L2 regularization:

```
L = -(1/n) Σ y_i * log(ŷ_i) + λ * Σ ||W||²
```

---

## 📈 Results & Performance

### Typical Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | ~97-98% |
| Training Time (50 epochs) | ~2-3 minutes |
| Best Architecture | 128→64→10 with ReLU |
| Optimal Learning Rate | 0.003 |

### Model Insights

✓ **ReLU** converges faster than ELU/Leaky ReLU  
✓ **Deeper networks** improve accuracy but increase training time  
✓ **Batch normalization** equivalent achieved through normalization in preprocessing  
✓ **Adam optimizer** outperforms vanilla SGD  
✓ Model struggles with similar digits (5↔8, 3↔8) — expected given dataset ambiguity  

### Visualizations Included

- **Loss Curves** — Monitor convergence during training
- **Accuracy Tracking** — Epoch-by-epoch performance
- **Confusion Matrices** — Per-class accuracy breakdown
- **Misclassified Samples** — Understand failure modes
- **Confidence Distributions** — Model uncertainty analysis

---

## 📚 How to Learn from This

### For Beginners
1. **Read `src/model.py`** — Understand the NeuralNetwork class structure
2. **Study forward()** — See how data flows through layers
3. **Study backward()** — Understand gradient computation
4. **Run with small dataset** — Train on subset to see convergence quickly

### For Intermediate
1. **Modify activation functions** — Try implementing your own
2. **Experiment with architectures** — Use grid search to find optimal configs
3. **Add regularization techniques** — Dropout, batch norm, etc.
4. **Analyze gradients** — Check gradient values during training

### For Advanced
1. **Implement momentum variants** — RMSprop, Adagrad, etc.
2. **Add convolutional layers** — Move toward CNNs
3. **Profile performance** — Optimize matrix operations
4. **Extend to other datasets** — CIFAR-10, Fashion-MNIST, etc.

---

## 🎨 Interactive Features

### Draw & Predict

The Streamlit app provides a canvas where you can:
- ✏️ Draw handwritten digits (0-9)
- 📊 See confidence scores for all predictions
- 🎯 View processed input (preprocessing pipeline)
- 🧹 Clear and retry

```bash
streamlit run src/app.py
```

### Experiment Dashboard

Visual comparison across all runs:
- 📈 Accuracy/loss curves overlay
- 🏆 Ranked model performance
- ⏱️ Training time analysis
- 🔧 Configuration comparison

```bash
streamlit run src/dashboard.py
```

---

## 🔧 Customization

### Change Architecture

```bash
python src/main.py --mode single \
  --layers 256 128 64 10 \
  --acts relu relu relu linear \
  --lr 0.001 \
  --epochs 100 \
  --batch_size 32
```

### Grid Search Parameters

Edit `src/experiment.py` to define custom search spaces:

```python
LAYER_CONFIGS = [
    [128, 64, 10],
    [256, 128, 10],
    [128, 128, 64, 10],
    # Add more...
]

ACTIVATIONS = [
    ["relu", "relu", "linear"],
    ["relu", "leaky_relu", "linear"],
    # Add more...
]
```

### Save & Load Models

```python
from model import NeuralNetwork

# After training
model.save_model("models/my_model.npz")

# Later, load it
loaded_model = NeuralNetwork.load_model(
    "models/my_model.npz",
    X_dummy, Y_dummy
)
```

---

## 📋 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.23 | Core computations |
| Pandas | ≥1.5 | Data handling |
| Matplotlib | ≥3.7 | Visualizations |
| Seaborn | ≥0.12 | Enhanced plots |
| scikit-learn | ≥1.3 | Evaluation metrics |
| Streamlit | ≥1.30 | Interactive UI |
| Pillow | ≥10.0 | Image processing |

```bash
pip install -r requirements.txt
```

---

## 🧪 Reproducibility

All runs are reproducible with fixed seeds:

```python
np.random.seed(42)  # Weights initialization
```

Every experiment captures:
- Model configuration
- Hyperparameters
- Training metrics
- Loss/accuracy curves
- Final model weights

Located in `results/run_XXX/` for full auditability.

---

## 🎓 Key Learnings

This project teaches:

### Mathematical Foundations
- Matrix calculus and chain rule application
- Softmax and cross-entropy loss
- Gradient-based optimization
- Convex and non-convex optimization landscapes

### Implementation Skills
- Efficient NumPy operations for numerical computing
- Debugging neural networks (gradient checking, etc.)
- Memory management for large datasets
- Version control for reproducibility

### Best Practices
- Modular code architecture
- Comprehensive testing and validation
- Visualization for understanding
- Documentation and comments
- Experiment tracking and comparison

---

## 🚀 Future Enhancements

- 🔹 **Convolutional Layers** — Add CNN implementation
- 🔹 **Batch Normalization** — Improve training stability
- 🔹 **Dropout & Regularization** — Advanced generalization techniques
- 🔹 **Distributed Training** — Multi-GPU support
- 🔹 **Alternative Datasets** — CIFAR-10, Fashion-MNIST, custom data
- 🔹 **Model Export** — ONNX format for deployment
- 🔹 **Web API** — Flask/FastAPI endpoint for predictions
- 🔹 **Quantization** — Model compression for mobile

---

## 💡 Tips for Success

### Training Best Practices
✓ Start with small learning rates and increase gradually  
✓ Monitor both training and validation loss  
✓ Use early stopping to prevent overfitting  
✓ Normalize inputs to [0, 1] range  
✓ Shuffle data between epochs  

### Debugging
✓ Check for NaN/Inf values in gradients  
✓ Visualize feature distributions  
✓ Compare predictions on known samples  
✓ Test with small dataset first  
✓ Print intermediate values in forward/backward  

### Optimization
✓ Increase batch size for speed (if memory allows)  
✓ Experiment with different activation functions  
✓ Try different optimizers (Adam, SGD, RMSprop)  
✓ Profile code to find bottlenecks  

---

## 🤝 Contributing

Contributions are welcome! Ideas:
- 🔹 Add new activation functions
- 🔹 Implement alternative optimizers
- 🔹 Improve performance/efficiency
- 🔹 Add more evaluation metrics
- 🔹 Enhance visualizations
- 🔹 Fix bugs or improve documentation

**Fork → Create Branch → Commit → Push → Pull Request**

---

## 📖 References

### Papers & Books
- **"Neural Networks and Deep Learning"** — Michael Nielsen  
- **"Backpropagation and Differentiation"** — LeCun et al.  
- **"Adam: A Method for Stochastic Optimization"** — Kingma & Ba  

### Related Projects
- [TensorFlow](https://tensorflow.org) — Production ML framework
- [PyTorch](https://pytorch.org) — Research-focused framework
- [Fast.ai](https://fast.ai) — Practical deep learning courses

---

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

---

## 🌟 Support

If this project helped you learn or was useful, please consider:
- ⭐ **Starring** the repository
- 🔗 **Sharing** with others
- 💬 **Opening issues** for questions or bugs
- 🤝 **Contributing** improvements

---

## 📬 Questions?

Feel free to open an issue or reach out directly. Questions about the implementation, architecture, or learning concepts are always welcome!

---

<div align="center">

**Built with ❤️ for learning neural networks from first principles**

*"The best way to understand neural networks is to implement them yourself."*

</div>