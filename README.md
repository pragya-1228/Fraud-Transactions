# 💳 Fraud Detection for Blocker Fraud Company

This project is a Data Science & Machine Learning internship solution for the **Blocker Fraud Company**, which specializes in detecting fraudulent mobile financial transactions. The model was designed to maximize business value using high-performance machine learning techniques and risk-based cost modeling.

---

## 📌 Business Problem

Blocker Fraud's business model is performance-based:
- ✅ **25% of the value** of each correctly identified fraud.
- ⚠️ **5% of the value** of each legitimate transaction incorrectly flagged as fraud.
- ❌ **100% reimbursement** for each fraudulent transaction incorrectly marked as legitimate.

Due to this high-risk, high-reward model, the **accuracy and precision** of the model directly impact the company’s profitability.

---

## 🎯 Objective

Develop a predictive system that:
- Detects fraudulent transactions with high precision and recall.
- Minimizes financial losses due to false negatives.
- Simulates business revenue and loss scenarios.

---

## 📂 Dataset Overview

- 📦 File: `Fraud.csv`
- 📊 Shape: 6,362,620 rows × 10 columns
- 🎯 Target: `isFraud` (1 = fraud, 0 = not fraud)

### ✍️ Columns Summary
| Column Name        | Description                              |
|--------------------|------------------------------------------|
| `step`             | Hour-wise time unit                      |
| `type`             | Transaction type                         |
| `amount`           | Amount of transaction                    |
| `oldbalanceOrg`    | Sender's balance before transaction      |
| `newbalanceOrig`   | Sender's balance after transaction       |
| `oldbalanceDest`   | Receiver's balance before transaction    |
| `newbalanceDest`   | Receiver's balance after transaction     |
| `isFraud`          | Whether transaction was fraud            |
| `isFlaggedFraud`   | Internally flagged fraud (rule-based)    |

---

## 🧪 Methodology

### 1. Data Description
- Checked and handled missing values
- Analyzed statistical metrics: mean, median, skewness, kurtosis, etc.

### 2. Feature Engineering
- Generated hypotheses using mind mapping
- Created new derived features to improve prediction

### 3. Data Filtering
- Removed non-informative and unrealistic records (e.g., out-of-range ages)

### 4. Exploratory Data Analysis
- Performed univariate, bivariate, and multivariate analysis
- Validated hypotheses:
  - ✅ All frauds are above ₹10,000
  - ❌ Fraud does **not** occur 60% of the time in CASH_OUT
  - ❌ High-value transactions are not limited to TRANSFER types

### 5. Data Preparation
- Applied label encoding
- Scaled features
- Handled class imbalance

### 6. Feature Selection
- Used the **Boruta algorithm** to select relevant features
- Reduced overfitting and improved interpretability

### 7. Model Building
- Trained various models with cross-validation:
  - Logistic Regression
  - K-Nearest Neighbors
  - Random Forest
  - **XGBoost** (Best Performing)
  - LightGBM
  - Dummy Classifier (baseline)

### 8. Hyperparameter Tuning
- Applied fine-tuning on the best model (XGBoost) for optimal performance

### 9. Conclusion
- Evaluated the generalization capacity on unseen test data

### 10. Model Deployment
- Model and functions saved
- **Flask API** created for production use (deployment plan on Heroku)

---

## 📊 Cross-Validation Results

| Model                  | Balanced Accuracy | Precision        | Recall           | F1 Score         | Kappa           |
|------------------------|-------------------|------------------|------------------|------------------|-----------------|
| Dummy                  | 0.500 ± 0.000      | 0.000 ± 0.000     | 0.000 ± 0.000     | 0.000 ± 0.000     | 0.000 ± 0.000    |
| Logistic Regression    | 0.654 ± 0.002      | 0.953 ± 0.007     | 0.309 ± 0.005     | 0.466 ± 0.005     | 0.466 ± 0.005    |
| K Nearest Neighbors    | 0.801 ± 0.006      | 0.949 ± 0.005     | 0.602 ± 0.011     | 0.736 ± 0.008     | 0.736 ± 0.008    |
| Support Vector Machine | 0.595 ± 0.013      | 1.000 ± 0.000     | 0.190 ± 0.026     | 0.319 ± 0.037     | 0.319 ± 0.037    |
| Random Forest          | 0.896 ± 0.006      | 0.970 ± 0.003     | 0.793 ± 0.011     | 0.872 ± 0.007     | 0.872 ± 0.007    |
| XGBoost                | 0.919 ± 0.006      | 0.955 ± 0.006     | 0.839 ± 0.012     | 0.893 ± 0.009     | 0.893 ± 0.009    |
| LightGBM               | 0.698 ± 0.123      | 0.329 ± 0.260     | 0.404 ± 0.238     | 0.345 ± 0.265     | 0.344 ± 0.266    |



## 🏆 Best Model: XGBoost

The **XGBoost classifier** was the best-performing model based on cross-validation and test set performance. After hyperparameter tuning, it achieved strong metrics on both training and unseen data.

### 🔁 Cross-Validation Performance

| Metric             | Score (± Std Dev)    |
|--------------------|----------------------|
| Balanced Accuracy  | 0.919 ± 0.006        |
| Precision          | 0.955 ± 0.006        |
| Recall             | 0.839 ± 0.012        |
| F1 Score           | 0.893 ± 0.009        |
| Cohen’s Kappa      | 0.893 ± 0.009        |

---

### 🧪 Test Set (Unseen Data) Performance

| Metric             | Score                |
|--------------------|----------------------|
| Balanced Accuracy  | 0.915                |
| Precision          | 0.944                |
| Recall             | 0.829                |
| F1 Score           | 0.883                |
| Cohen’s Kappa      | 0.883                |

---

### ✅ Highlights
- Successfully captured **~83% of fraudulent transactions** in unseen data.
- Achieved **high precision** (~95%) reducing the rate of false positives.
- Balanced accuracy shows robustness against extreme class imbalance.




---

## 📊 Business Simulation Results

| Scenario                                | Amount (R$)         |
|-----------------------------------------|----------------------|
| ✅ Revenue from correct fraud detection  | 60,613,782.88        |
| ⚠️ Revenue from false positives           | 183,866.98           |
| ❌ Refund due to false negatives         | -3,546,075.42        |
| 💰 **Net Profit using model**            | **57,251,574.44**    |
| ❌ Profit using existing method          | **-246,001,206.94**  |

---

## 📚 Lessons Learned
- Models can be accurate even with extreme class imbalance (<1% fraud)
- Hypothesis testing strengthens feature engineering
- Business cost modeling is vital in high-risk industries like fraud detection

---

## 🛠️ Tech Stack

- Python 3.8+
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn, XGBoost, LightGBM, BorutaPy
- Flask (for deployment)
- Jupyter Notebook

---

## 🧪 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/blocker-fraud-detection.git
cd blocker-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Jupyter Notebook
jupyter notebook

# 4. (Optional) Run the Flask API
python app.py
