🏦 Loan Eligibility Prediction App

📌 Project Overview
This project aims to automate the loan eligibility prediction process for a financial institution using machine learning. Based on user-provided details such as gender, income, education, credit history, and other features, the app predicts whether a loan applicant is likely to be eligible for a loan.

Built using Streamlit, this interactive web app enables real-time predictions based on trained models.

## 🗂️ Dataset Overview

Among all industries, the **insurance domain** has one of the largest uses of analytics and data science. This dataset simulates a real-world insurance loan scenario. It contains **615 rows and 13 columns**.

### 📌 Problem Statement

The company wants to **automate the loan eligibility process** in real-time based on customer details submitted in an online loan application form. The data includes variables such as:

- Gender
- Marital Status
- Education
- Number of Dependents
- Applicant Income
- Loan Amount
- Credit History
- Property Area
- And others

The task is to build a classification model that can **identify customer segments** who are eligible for a loan, enabling the business to efficiently target these individuals.

---

### 🔍 Challenges Addressed

- Handling missing values
- Encoding categorical features
- Feature scaling (only where necessary)
- Training and evaluating multiple ML models
- Hyperparameter tuning for model optimization

## 🚀 Objectives

- Train and evaluate multiple classification models
- Perform hyperparameter tuning with `GridSearchCV`
- Select the best model based on evaluation metrics
- Visualize confusion matrices and ROC curves
- Deploy the model using Streamlit (optional)

---

## 🧪 Models Trained

| Model                 | Scaled? | Tuned? | Accuracy |
|----------------------|---------|--------|----------|
| Logistic Regression   | ✅      | ✅     | 78.86%   |
| Decision Tree         | ❌      | ❌     | 70.73%    |
| Naive Bayes           | ✅      | ✅     | 78.05%    |
| Random Forest         | ❌      | ✅     | ✅ **Best Model** (76.42%) |

🔍 After Hyperparameter Tuning
| Model               | Best Params Found                                                                      | Accuracy |
| ------------------- | -------------------------------------------------------------------------------------- | -------- |
| Logistic Regression | `{'C': 0.01, 'solver': 'liblinear'}`                                                   | 78.86%   |
| Random Forest       | `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}` | 78.05%   |
| Naive Bayes         | `{'var_smoothing': 1e-09}`                                                             | 78.05%   |


✅ Final Model Selection: **Random Forest Classifier**
Although Logistic Regression had a slightly higher accuracy (78.86%), Random Forest was chosen as the final model for deployment because of the following reasons:

🔁 Robust to Overfitting due to ensemble averaging.

🌳 Captures Non-linear Relationships better than linear models.

📊 Provides Feature Importance, enhancing interpretability.

💪 Stronger Generalization across unseen data in cross-validation.

---

## 🔍 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC AUC Score
- ROC Curve

---

## 📊 Visualizations

- Confusion matrices for all models  
- ROC Curve for the best model  
- Bar chart comparison of model accuracies

---

## 🛠️ Technologies Used

- Python 3.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Streamlit (for deployment)

---

## 💾 Model Saving

The final model (`RandomForestClassifier`) is saved using `joblib` or `pickle` for future predictions.

---

## 🎯 Final Results

- **Best Accuracy**: 78.05%
- **Selected Model**: `Random Forest`
- **Tested on Unseen Data**: ✅

---
🧠 Key Learnings
Compared multiple ML algorithms: Logistic Regression, Random Forest, Naive Bayes, Decision Tree.

Implemented hyperparameter tuning using GridSearchCV.

Evaluated models using Accuracy, Confusion Matrix, and ROC-AUC.

Built a user-friendly prediction interface with Streamlit.
