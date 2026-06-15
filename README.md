# diabetes-detection-tensorflow
# Diabetes Detection using TensorFlow

## 📌 Project Overview

This project implements a **binary classification model** using TensorFlow to predict whether a patient has diabetes based on diagnostic measurements from the Pima Indians Diabetes Dataset.

Two models were developed and compared:

1. **TensorFlow Classifier without Hidden Layers**
2. **TensorFlow Classifier with One Hidden Layer and One Dropout Layer**

The objective is to analyze the impact of adding hidden layers and dropout regularization on model performance.

---

## 📊 Dataset

Dataset: Pima Indians Diabetes Dataset

Features:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

Target Variable:

* Outcome

  * 0 → Non-Diabetic
  * 1 → Diabetic

---

## 🔍 Exploratory Data Analysis (EDA)

Performed the following analyses:

* Dataset structure inspection
* Statistical summary
* Missing value analysis
* Class distribution analysis
* Correlation heatmap
* Histograms for feature distributions
* Boxplots for outlier detection
* Scatter plots for feature relationships
* Pairplot visualization

Key Findings:

* Glucose showed the strongest correlation with diabetes.
* BMI and Age were important predictive factors.
* Some columns contained zero values that represented missing data.
* Outliers were observed in Insulin and BMI.

---

## ⚙️ Data Preprocessing

* Removed invalid zero values
* Handled missing values using median imputation
* Feature scaling using StandardScaler
* Train-Test Split (80%-20%)

---

## 🧠 Model 1: Binary Classifier Without Hidden Layer

Architecture:

Input Layer → Output Layer (Sigmoid)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

Characteristics:

* Simple Logistic Regression implemented in TensorFlow
* Suitable baseline model
* Fast training

---

## 🧠 Model 2: Binary Classifier with Hidden Layer and Dropout

Architecture:

Input Layer → Hidden Layer (ReLU) → Dropout (0.2) → Output Layer (Sigmoid)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

Characteristics:

* Learns non-linear relationships
* Better generalization
* Reduced overfitting through dropout

---

## 📈 Evaluation Metrics

* Accuracy
* Binary Cross Entropy Loss

Typical Results:

| Model                  | Test Accuracy |
| ---------------------- | ------------- |
| Without Hidden Layer   | ~75% - 78%    |
| Hidden Layer + Dropout | ~78% - 82%    |

Results may vary depending on train-test split and random initialization.

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## 📂 Project Structure

```text
diabetes-detection-tensorflow/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── diabetes_classification.ipynb
│
├── images/
│   ├── correlation_heatmap.png
│   ├── class_distribution.png
│   └── training_accuracy.png
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/diabetes-detection-tensorflow.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook or Google Colab notebook.

4. Run all cells.

---

## 🎯 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature engineering
* Model deployment using Flask or Streamlit
* Comparison with Random Forest and XGBoost

---

## 👩‍💻 Author

Satwika Alluri

Aspiring Data Scientist | Data Analyst | AI & ML Enthusiast

LinkedIn: https://www.linkedin.com/in/satwikaalluri123
