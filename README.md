# OIBSIP_TASK2_LEVEL2
💳 Credit Card Fraud Detection using Machine Learning
📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is a critical real-world application of data science that involves identifying rare and anomalous patterns in financial data. The goal is to build models that can distinguish between legitimate and fraudulent transactions with high accuracy and reliability.

🎯 Objectives

Analyze credit card transaction data

Handle highly imbalanced datasets

Detect anomalies and fraudulent patterns

Build and evaluate classification models

Improve fraud detection accuracy using preprocessing techniques

📂 Dataset

The dataset contains anonymized credit card transactions made by European cardholders.

Features:

V1 – V28 → PCA-transformed features (for privacy)

Time → Time elapsed between transactions

Amount → Transaction value

Class → Target variable

0 = Legitimate transaction

1 = Fraudulent transaction

🛠️ Technologies Used

Python

Pandas & NumPy (data processing)

Matplotlib & Seaborn (visualization)

Scikit-learn (machine learning)

⚠️ Challenges in Fraud Detection

Extreme class imbalance (fraud cases are very rare)

High cost of misclassification

Need for precision and recall over accuracy

Handling large-scale transactional data

📊 Exploratory Data Analysis

EDA was performed to:

Understand fraud distribution

Analyze class imbalance

Visualize feature relationships

Identify anomalies in transaction patterns

Key observation:

Fraud transactions form a very small percentage of total data.

🧠 Machine Learning Models Used
1️⃣ Logistic Regression

Baseline classification model

Works well for binary classification

Used with feature scaling and class balancing

2️⃣ Decision Tree Classifier

Captures non-linear patterns

Easy to interpret

Useful for fraud pattern detection

⚙️ Project Workflow

Data loading and inspection

Data preprocessing and scaling

Handling class imbalance

Train-test split

Model training

Model evaluation

📈 Evaluation Metrics

Due to class imbalance, multiple metrics were used:

Accuracy

Precision

Recall (important for fraud detection)

F1-score

Confusion Matrix

ROC–AUC Score

Recall was prioritized to minimize missed fraud cases.

📊 Results

The models successfully identified fraudulent transactions despite severe class imbalance. Logistic Regression provided stable and balanced results, while Decision Trees captured non-linear fraud patterns effectively. The project demonstrates how machine learning can be applied to real-world financial security problems.

🚀 Future Improvements

SMOTE for oversampling minority class

Random Forest / XGBoost implementation

Real-time fraud detection pipeline

Deep learning models (Neural Networks)

Deployment using Flask or Streamlit

▶️ How to Run
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook or script
python fraud_detection.py

📚 Learning Outcomes

Handling imbalanced datasets

Feature scaling and preprocessing

Model evaluation beyond accuracy

Practical understanding of anomaly detection

Real-world application of machine learning
