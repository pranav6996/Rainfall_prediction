
# 🌧️ Rainfall Prediction using Machine Learning

A supervised machine learning project to predict whether it will rain tomorrow using real-world weather data. This project explores feature engineering, model training, evaluation, and performance comparison between classifiers like Random Forest and Logistic Regression.

---

## 📁 Dataset

- **Source**: Australian Bureau of Meteorology  
- **Features**: 20+ including temperature, humidity, wind speed, etc.  
- **Target**: `RainTomorrow` — (Yes/No)  

---

## 🧪 Objective

> Predict whether it will rain the next day using weather data from today.

---

## 🧰 Tech Stack

- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## 🧼 Data Preprocessing

- **Missing Values**  
  - Numerical: Imputed with median  
  - Categorical: Imputed with most frequent  

- **Encoding**  
  - OneHotEncoding for nominal categorical features  
  - Standard Scaling for numerical features  

- **Train-Test Split**  
  - Commonly 80/20 split

---

## 🏗️ Pipeline Construction

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

- Combined numerical and categorical transformations into a single pipeline  
- Used `ColumnTransformer` to apply preprocessing correctly to different feature types  

---

## 🤖 Models Used

1. **Random Forest Classifier**
   - Tuned via `GridSearchCV`
   - Used feature importances for interpretability

2. **Logistic Regression**
   - Solver: `liblinear`
   - Penalized with L1/L2 regularization

---

## 🧮 Evaluation Metrics

- **Accuracy**
- **Precision / Recall / F1 Score**
- **Confusion Matrix**
- **True Positive Rate (TPR)**

---

### 📊 Example Confusion Matrix (Random Forest)

> ✅ **True Positive Rate (TPR)**:  
> TPR = TP / (TP + FN) = 183 / (183 + 175) ≈ **51%**

---

## 🧬 Feature Importance (Random Forest)

Top features

## 🔍 Interpretation

- **Random Forest**:
  - Higher accuracy and recall
  - Handles non-linearity and feature interactions well  
- **Logistic Regression**:
  - More interpretable
  - Simpler, but lower TPR

> 🏆 **Winner**: Random Forest — better generalization and true positive rate.

---

## 📌 Key Learnings

- Feature engineering is **critical** in real-world data projects
- There's always a trade-off:  
  - Accuracy vs Interpretability  
  - Simplicity vs Performance  
- Real datasets come with:  
  - Missing values  
  - Imbalanced classes  
  - Noisy features  

---

## 🧠 Future Work

- Try ensemble models: `XGBoost`, `LightGBM`
- Balance classes using:
  - SMOTE
  - Class weights
- Add regression to predict **actual rainfall amount**, not just Yes/No
