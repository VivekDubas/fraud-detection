# 🕵️‍♂️ AI-Powered Fraud Detection System

This project is developed for a fraud detection hackathon challenge. The goal is to detect **fraudulent financial transactions** using machine learning and deep learning, while handling class imbalance effectively.

---

## 📌 Problem Statement

> Build a real-time fraud detection model using historical transaction data. The model should be optimized for **precision**, **recall**, **F1-score**, and **AUC-ROC**, especially for the minority class (fraudulent transactions).

---

## 📊 Dataset Insights

- **Binary Classification**:  
  - 0 → Genuine transaction  
  - 1 → Fraudulent transaction

- **Severe Class Imbalance**:  
  - Fraudulent transactions are less than 1.3% of the data.

---

## 🧪 Exploratory Data Analysis (EDA)

Key insights:
- Transactions with **gender = 'U'** are always genuine.
- **Merchant categories like `es_leisure`** show high fraud density.
- **Mismatched ZIP codes** between customer and merchant hint at fraud.
- Certain **high-value transactions** are strongly correlated with fraud.

---

## ⚙️ Preprocessing

- **Label Encoding / One-Hot Encoding** for categorical features (`gender`, `category`, etc.)
- **StandardScaler** for numerical columns (`amount`, `step`, `age`)
- Added feature: **log(amount)** to reduce skew
- Added feature: **autoencoder reconstruction error** for anomaly signal

---

## ⚠️ Addressing Class Imbalance

Class imbalance was tackled using multiple strategies:

### ✅ Approaches Tried:

| Approach                      | Result Summary                                |
|------------------------------|-----------------------------------------------|
| **Focal Loss (α=0.25, γ=2)** | Boosts focus on hard-to-classify frauds       |
| `class_weight` in MLP        | Assigns higher loss weight to frauds          |
| `RandomOverSampler`          | Duplicates minority samples to balance        |
| `SMOTE`                      | Synthesizes realistic minority samples        |
| `SMOTEENN`                   | Hybrid of oversampling + cleaning noise (slow & less effective in this case) |

✅ Final best results used: **SMOTE + Focal Loss**

---

## 🤖 Final Model Architecture

Implemented using **TensorFlow Keras**:

python
Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

### 🔍 Model Highlights

| 🔧 Component        | 💡 Description                                     |
|---------------------|---------------------------------------------------|
| **Input Layer**     | Takes preprocessed features                       |
| **Hidden Layers**   | ReLU activations with Dropout for regularization  |
| **Output Layer**    | Sigmoid activation for binary classification      |
| **Loss Function**   | Custom **Focal Loss** to focus on hard examples   |
| **Optimizer**       | Adam optimizer                                    |
| **Callback**        | EarlyStopping to avoid overfitting                |

### 🎯 Focal Loss Function

Focal Loss is used to dynamically scale the contribution of easy vs hard examples during training, which is crucial for handling **imbalanced datasets** like fraud detection.

python
import tensorflow as tf
from tensorflow.keras import backend as K

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


 Evaluation Metrics
Metric	Value
Accuracy	~99.7%
Precision (1)	0.81
Recall (1)	0.82
F1-Score (1)	0.80
AUC-ROC Score	0.9978

Libraries Used
pandas, numpy

scikit-learn

tensorflow, keras

imblearn (SMOTE, RandomOverSampler, SMOTEENN)

matplotlib, seaborn (for EDA)

