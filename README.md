# Classifying-Handwritten-Digits-Using-SVMs

I built a handwritten digit classifier using SVM where I converted images into feature vectors, applied an RBF kernel for non-linear separation, and achieved high accuracy in predicting digits from image data.

Details👇
# 👩‍💻 Classifying Handwritten Digits Using SVMs

## 📋 Project Overview

This project focuses on building a machine learning model to classify handwritten digits (0–9) using **Support Vector Machines (SVMs)**. The objective is to compare the performance of **Linear** and **RBF (Radial Basis Function)** kernels and understand how kernel selection impacts classification accuracy.

The dataset used is the **Digits dataset from Scikit-learn**, which contains grayscale images of handwritten digits.

---

## 🎯 Objectives

* Load and explore image-based numerical data
* Preprocess and normalize features
* Train SVM models using different kernels
* Evaluate model performance using multiple metrics
* Tune hyperparameters to improve accuracy
* Compare linear vs non-linear classification

---

## 📂 Dataset Description

* Source: `sklearn.datasets.load_digits()`
* Total samples: **1797**
* Image size: **8 × 8 pixels**
* Features per sample: **64 (flattened pixel values)**
* Target classes: **10 (digits 0–9)**

Each image is converted into a feature vector representing pixel intensity values.

---

## ⚙️ Project Workflow

### 🔹 1. Data Loading & Exploration

* Loaded dataset using Scikit-learn
* Extracted:

  * `X` → feature matrix (pixel values)
  * `y` → target labels (digits)
* Visualized sample images to understand input data

---

### 🔹 2. Train-Test Split

* Split dataset into:

  * **Training set (80%)**
  * **Testing set (20%)**
* Used **stratified sampling** to maintain class balance

---

### 🔹 3. Data Preprocessing

* Applied **StandardScaler** for normalization
* Important steps:

  * Fit scaler only on training data
  * Transform both training and testing data

👉 This ensures no data leakage and improves model performance.

---

### 🔹 4. Training SVM (Linear Kernel)

* Used:

  ```python
  SVC(kernel='linear')
  ```
* Linear kernel attempts to separate classes using a straight decision boundary

---

### 🔹 5. Evaluation of Linear SVM

* Metrics used:

  * Accuracy score
  * Classification report (Precision, Recall, F1-score)
  * Confusion matrix

📊 Result:

* Achieved around **~95% accuracy**
* Some digits were misclassified due to linear limitations

---

### 🔹 6. Training SVM (RBF Kernel)

* Used:

  ```python
  SVC(kernel='rbf')
  ```
* RBF kernel maps data into higher dimensions for better separation

---

### 🔹 7. Evaluation & Comparison

* Same evaluation metrics applied

📊 Result:

* Accuracy improved to **~98%+**
* Better separation of complex digit patterns
* Reduced misclassification compared to linear model

---

### 🔹 8. Hyperparameter Tuning

Used **GridSearchCV** to optimize:

* `C` (Regularization parameter)
* `gamma` (Kernel coefficient)

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}
```

✔ Best model selected based on cross-validation performance
✔ Final model evaluated on test data

---

## 📊 Model Performance Summary

| Model         | Accuracy | Key Insight                                   |
| ------------- | -------- | --------------------------------------------- |
| Linear SVM    | ~95%     | Works well but limited for complex patterns   |
| RBF SVM       | ~98%+    | Captures non-linear relationships effectively |
| Tuned RBF SVM | Highest  | Optimized performance with hyperparameters    |

---

## 🔍 Key Insights

* **Kernel choice is critical** in SVM performance
* Linear kernel struggles with complex digit boundaries
* RBF kernel handles non-linear relationships effectively
* Feature scaling is essential for SVM performance
* Hyperparameter tuning significantly improves accuracy

---

## 📈 Visualization

* Displayed sample digit images
* Used confusion matrices to analyze prediction errors
* Compared misclassification patterns across models

---

## 🚀 Conclusion

This project demonstrates how SVMs can be effectively used for image classification tasks. While linear models provide a good baseline, non-linear kernels like RBF significantly enhance performance by capturing complex patterns in the data.

---

## 🧠 Future Improvements

* Use full **MNIST dataset (28×28 images)**
* Try **other models (CNN, Random Forest, KNN)**
* Implement **real-time digit recognition**
* Optimize performance using advanced tuning techniques

---

## 🛠️ Tech Stack

* Python
* NumPy, Pandas
* Matplotlib
* Scikit-learn

---

## 📌 How to Run

1. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Run the Python script:

   ```bash
   python svm_digits.py
   ```

