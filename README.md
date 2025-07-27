# 🕵️‍♂️ AI-Powered Fraud Detection System

A powerful machine learning and deep learning–based solution developed for a **fraud detection hackathon** challenge. The objective: **detect fraudulent financial transactions** with high precision and recall, especially in the presence of **severe class imbalance**.

## 📌 Problem Statement

Build a real-time fraud detection model using historical transaction data. Optimize the model for **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**, particularly for the **minority class** (fraudulent transactions).

## 📊 Dataset Overview

* **Type**: Binary Classification
  * `0` → Genuine Transaction
  * `1` → Fraudulent Transaction
* **Challenge**:
  * ⚠️ Fraudulent transactions are **< 1.3%** of the data (extremely imbalanced)

## 🧪 Exploratory Data Analysis (EDA)

Key insights derived from the data:

* Transactions with `gender = 'U'` were always genuine.
* Merchant category `es_leisure` exhibited high fraud concentration.
* **Mismatched ZIP codes** between customer and merchant were common in frauds.
* Certain **high-value transactions** strongly correlated with fraud labels.

## ⚙️ Preprocessing Pipeline

* ✅ **Categorical Encoding**:
  * Label Encoding / One-Hot Encoding on `gender`, `category`, etc.
* ✅ **Numerical Scaling**:
  * StandardScaler used on `amount`, `step`, `age`
* ✅ **Feature Engineering**:
  * `log(amount)` to handle skew
  * Autoencoder-based **reconstruction error** added as an anomaly feature

## ⚠️ Tackling Class Imbalance

To handle class imbalance, multiple advanced techniques were implemented and evaluated:

| Technique | Description | Outcome |
|-----------|-------------|---------|
| 🔥 **Focal Loss** | Loss function emphasizing hard samples (`α=0.25`, `γ=2`) | Improved minority class learning |
| ⚖️ `class_weight` | Higher loss weight to class `1` (frauds) | Mild improvement |
| 🔁 `RandomOverSampler` | Duplicates minority samples | Fast and simple, decent improvement |
| 🧬 `SMOTE` | Synthesizes new samples for minority class | Best result with Focal Loss |
| ⚗️ `SMOTEENN` | SMOTE + Edited Nearest Neighbors (noise removal) | Too slow, and not effective for this dataset |

✅ **Best Strategy**: **SMOTE + Focal Loss**

## 🔧 Autoencoder Architecture

First, an autoencoder was built to generate reconstruction error features for anomaly detection:

```python
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
bottleneck = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)
```

The reconstruction error from this autoencoder was used as an additional feature to help identify anomalous transactions.

## 🤖 Final Deep Learning Model

Built using **TensorFlow Keras**:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 🔍 Model Details

| Component | Description |
|-----------|-------------|
| Input Layer | Takes in the preprocessed features |
| Hidden Layers | ReLU activations with Dropout regularization |
| Output Layer | Sigmoid activation for binary classification |
| Loss Function | **Custom Focal Loss** to focus on minority class |
| Optimizer | Adam |
| Callback | EarlyStopping for generalization |

## 🧠 Focal Loss Function

```python
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + \
                (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        
        return K.mean(weight * cross_entropy)
    
    return focal_loss_fixed
```

## 📈 Final Evaluation Metrics

| Metric | Value |
|--------|-------|
| ✅ Accuracy | ~99.7% |
| ✅ Precision (Fraud) | 0.81 |
| ✅ Recall (Fraud) | 0.82 |
| ✅ F1-Score (Fraud) | 0.80 |
| ✅ AUC-ROC Score | **0.9978** |

## 📚 Libraries Used

* `pandas`, `numpy` – Data processing
* `scikit-learn` – Preprocessing, metrics, models
* `tensorflow`, `keras` – Deep learning
* `imblearn` – SMOTE, RandomOverSampler, SMOTEENN
* `matplotlib`, `seaborn` – Visualization & EDA

## 🚀 Project Highlights

* ✔️ Strong **fraud recall and precision** despite imbalance
* ✔️ Combined **autoencoder anomaly detection** and **MLP**
* ✔️ Evaluated various **resampling + loss** strategies
* ✔️ Deployable and interpretable solution

## 🛠️ Installation & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn imbalanced-learn
```

### Quick Start

```python
# Load and preprocess data
from preprocessing import load_and_preprocess_data
X_train, X_test, y_train, y_test = load_and_preprocess_data('fraud_data.csv')

# Apply SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train model with Focal Loss
from model import create_fraud_detection_model, focal_loss
model = create_fraud_detection_model(input_dim=X_train.shape[1])
model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
model.fit(X_train_smote, y_train_smote, validation_split=0.2, epochs=50)

# Evaluate
predictions = model.predict(X_test)
```

## 📁 Project Structure

```
fraud-detection/
├── data/
│   └── fraud_data.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## 🏆 Results Summary

This fraud detection system successfully addresses the challenge of highly imbalanced financial transaction data by combining advanced sampling techniques with custom loss functions. The model achieves excellent performance metrics while maintaining practical deployability for real-world fraud detection scenarios.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities, please reach out via GitHub issues.
