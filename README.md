## Basic Regression Neural Network (Iris-derived dataset)

This repository contains a small PyTorch-based regression neural network implemented in a Jupyter notebook (`iris_nn.ipynb`) using a custom Iris-like CSV dataset (`iris_data.csv`). The model predicts three continuous target scores per sample (SetosaScore, VersicolorScore, VirginicaScore) from the four classic Iris features.

### Contents
- `iris_nn.ipynb` — the notebook with data loading, preprocessing, model definition, training loop, evaluation and example predictions.
- `iris_data.csv` — the CSV dataset used by the notebook. Columns: `Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `SetosaScore`, `VersicolorScore`, `VirginicaScore`.
- `requirements.txt` — minimal Python dependencies to run the notebook.

### Model summary
- Framework: PyTorch
- Input: 4 features (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
- Output: 3 regression targets (SetosaScore, VersicolorScore, VirginicaScore)
- Architecture: Sequential linear layers (no activations)
  - fc1: Linear(4, 128)
  - fc2: Linear(128, 256)
  - fc3: Linear(256, 512)
  - fc4: Linear(512, 3)
- Loss: Mean Absolute Error (L1Loss)
- Optimizer: Adam (learning rate 0.001)

Notes: the notebook intentionally uses only linear layers (no activations) and no explicit regularization, per the notebook's instructions.

### Preprocessing & training details
- Train/test split: 80% training / 20% test (scikit-learn `train_test_split`, random_state=42)
- Feature scaling: `StandardScaler` applied to input features (fit on training set, applied to test set)
- Data batching: PyTorch `TensorDataset` + `DataLoader` with `batch_size=15` and `shuffle=True`
- Epochs: 300 (printed per-epoch average training loss in the notebook)

### How to run
1. Create a Python environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / zsh
pip install -r requirements.txt
```

2. Open the notebook and run the cells in order (Jupyter / VS Code notebook):

```bash
jupyter notebook iris_nn.ipynb
```

3. The notebook prints training progress (average loss per epoch) and a final test loss. It also shows example predictions for sample inputs.

### Example usage (what to expect)
- Training prints: `Epoch: <n>, Average Loss: <val>` for each epoch.
- After training the notebook prints: `Test loss: <val>`.
- Example prediction block prints three scores per sample:

```
Prediction 1:
- Setosa Score: 0.xxx
- Versicolor Score: 0.yyy
- Virginia Score: 0.zzz
```

### Suggestions / next steps
- Add non-linear activations (ReLU, GELU) between linear layers to increase expressive power.
- Add a validation split and early stopping to avoid overfitting.
- Try MSE loss for regression and compare results to MAE.
- Export training logs (TensorBoard or CSV) and plot loss curves.

### Contact
If you want me to convert the notebook into a standalone script, add training logs, or add a small test harness, tell me which option you'd like and I will implement it.
