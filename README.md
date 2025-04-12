# 🌲 Land Cover Classification Using Spectral Data

## 📘 Project Overview

This project explores **land cover classification** based on spectral data collected via aerial sensors. The goal is to build and compare the performance of multiple classification algorithms that categorize land types (e.g., Sugi forest, Hinoki forest, Mixed Deciduous forest, and Other land types).

The project emphasizes:
- Preprocessing spectral data
- Applying and tuning multiple classifiers
- Evaluating performance using precision, recall, F1-score, and confusion matrices
- Visualizing classification results and class distributions

---

## 🧠 What I Learned

Through this project, I developed hands-on experience in:

- 📊 **Data standardization** and visual inspection using boxplots and histograms
- 🔍 **Model selection and hyperparameter tuning** using validation sets
- ⚙️ Implementing and optimizing:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVMs) using One-vs-Rest and One-vs-One
  - Random Forest Classifier
- 🧪 Performance evaluation using:
  - Confusion matrices
  - F1-scores, precision, and recall
- 🧰 Using **manual grid search** for hyperparameter optimization
- 📈 Visual interpretation of classification model behavior

---

## 🛠️ Tools and Libraries Used

- `pandas`, `numpy` – Data loading and transformation
- `matplotlib`, `seaborn` – Visualization and exploration
- `sklearn` – Machine learning models, preprocessing, and metrics
- `statsmodels`, `scipy` – Statistical support and diagnostics

---

## ⚙️ Models Implemented

### ✅ K-Nearest Neighbors (KNN)
- Grid search over `k`, distance metric, and weight function
- Achieved best results with selected combination of hyperparameters

### ✅ Support Vector Machines (SVM)
- Explored both **One-vs-Rest** and **One-vs-One** classification schemes
- Tuning of `C`, `gamma`, and `class_weight` with linear kernel
- Model evaluated based on **macro-averaged F1-score**

### ✅ Random Forest Classifier
- Tuned `n_estimators`, `max_depth`, and `class_weight`
- Strong generalization performance and interpretability

---

## 📊 Evaluation Strategy

Models are evaluated using:
- **Confusion matrices** (normalized for interpretability)
- **F1-score** as the primary metric, especially macro-averaged
- Visual inspection of class distribution in training, validation, and test sets

---

## 🚀 How to Run

1. Clone the repository
2. Ensure you have the required libraries installed:

"pip install pandas matplotlib seaborn scikit-learn statsmodels"

3. Run the script or Jupyter notebook to train models and visualize evaluation results

---

## 🔎 Insights

- Proper **data preprocessing** (e.g., standardization) greatly improves model accuracy.
- SVM and Random Forests both provide strong performance, with trade-offs in training time and interpretability.
- **Hyperparameter tuning** significantly impacts the effectiveness of classifiers, especially for small or imbalanced datasets.

---

## 📬 Contact

**Mohan Hao**  
Data & ML Enthusiast  
📧 imhaom@gmail.com

---
