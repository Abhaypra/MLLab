# Deposify – Predictive Analytics for Term Deposit Subscriptions

🎯 **Deposify** is a machine learning-powered solution designed to predict whether a customer will subscribe to a bank's term deposit product. By analyzing historical data on customer demographics, financial behavior, and marketing campaign interactions, this system helps financial institutions run smarter, more efficient campaigns and optimize customer targeting.

---

## 🚀 Project Overview

In the financial industry, customer acquisition and retention are critical. Marketing campaigns are often broad and inefficient, leading to wasted resources. **Deposify** enables banks to predict which customers are likely to subscribe to term deposit schemes, improving conversion rates and reducing marketing costs.

---

## 🧠 Problem Statement

Traditional bank marketing campaigns suffer from:
- Low conversion rates
- High acquisition costs
- Poor targeting

**Objective:** Build a predictive model that classifies whether a customer will subscribe to a term deposit based on historical data.

---

## 🧾 Dataset

The project uses a dataset derived from real-world bank marketing campaigns. The dataset includes features such as:

- Personal: `age`, `job`, `marital`, `education`
- Financial: `balance`, `housing`, `loan`
- Campaign-related: `contact`, `month`, `day`, `duration`, `campaign`, `pdays`, `previous`
- Target Variable: `y` (yes/no - whether the customer subscribed)

- 📁 `train(1).csv` – Used to train and validate the model  
- 📁 `test.csv` – Used for final model inference and predictions

---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - `Pandas`, `NumPy` – Data handling
  - `Matplotlib`, `Seaborn` – Visualization
  - `Scikit-learn` – Machine learning models and evaluation
  - `Jupyter Notebook` – Experimentation and insights

---

## 🔍 Features & Capabilities

- Data Cleaning and Preprocessing
- Categorical Variable Encoding (LabelEncoder / OneHotEncoder)
- Feature Selection and Importance Evaluation
- Multiple Model Training and Comparison
- Performance Evaluation using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
- Final Prediction on Unseen Test Data

---

## 📈 Model Building

Trained and compared several classification algorithms:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Model performance was compared using evaluation metrics to choose the best-performing algorithm.

---

## 🧪 Evaluation Metrics

To ensure robustness and fairness, we evaluated models using:
- Accuracy Score
- Precision & Recall
- F1-Score
- Confusion Matrix

These metrics give insight into how well the model handles both majority and minority classes.

---

## 💡 Business Impact

Deposify offers measurable improvements in banking operations:
- 🎯 Increased targeting efficiency for marketing campaigns
- 📉 Reduction in customer acquisition cost
- 🤝 Better customer engagement through personalization
- 📈 Boost in term deposit subscription rates

---

## 📊 Visualizations

- Class Distribution
- Correlation Heatmaps
- Feature Importance Graphs
- Prediction Result Charts

(See notebook for visual examples.)

---

## 🛠 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/abhaypra/MLLab/deposify-term-deposit-prediction.git
   cd deposify-term-deposit-prediction


2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:

   ```bash
   jupyter notebook Bank_term_deposit_prediction.ipynb
   ```

4. Follow the steps in the notebook to view analysis, model training, and predictions.

---

## 📁 Folder Structure

```
Deposify/
├── Bank_term_deposit_prediction.ipynb
├── train(1).csv
├── test.csv
├── requirements.txt
└── README.md
```

---

## 📌 Future Improvements

* Hyperparameter tuning for improved accuracy
* Deployment using  Streamlit
* Incorporation of real-time data streams

