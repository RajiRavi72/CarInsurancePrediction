# 🚗 Car Insurance Claim Prediction

This project predicts whether a customer will make a **car insurance claim** in the next policy period, based on demographic, vehicle, and policy details.  
It also includes an interactive **Streamlit web app** for real-time prediction and a **Power BI dashboard** for visualization.

---

## 📊 Project Overview

- **Goal:** Predict the probability of a car insurance claim (`is_claim`) using machine learning.  
- **Dataset:** Provided `train.csv` and `test.csv` files with policy and customer data.  
- **Model:** Tuned **LightGBM Classifier** with preprocessing pipeline for numeric and categorical features.  
- **Challenge:** Strong class imbalance (very few claim cases), handled using `scale_pos_weight`.

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preprocessing
- Numeric and categorical features separated.
- Missing values handled.
- OneHotEncoding for categorical columns.
- Standard scaling for numeric columns.
- Saved as `preprocessor.pkl`.

### 2️⃣ Model Training
- Algorithm: **LightGBMClassifier**
- Class imbalance handled with `scale_pos_weight`.
- Evaluation metric: **AUC Score** and **Classification Report**.
- Final model saved as `lgbm_best_model.pkl`.

### 3️⃣ Prediction
- Input preprocessed using the same transformer.
- Outputs:
  - **Claim Probability (%).**
  - **Predicted Class (Claim / No Claim).**
- Created submission files:
  - `submission.csv`
  - `submission.xlsx` (for Kaggle-style submission).

---

## 🧰 Tech Stack

| Component | Tool/Library |
|------------|--------------|
| Programming Language | Python |
| Web App Framework | Streamlit |
| ML Framework | LightGBM |
| Preprocessing | Scikit-learn |
| Visualization | Power BI |
| Data Handling | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 🚀 How to Run the Streamlit App

### Step 1: Install Dependencies

pip install -r requirements.txt
streamlit run app.py
Step 3: Interact
Fill out customer and vehicle details.
Click Predict Claim Probability.
See claim probability and class instantly.

📈 Power BI Dashboard
After generating predictions (submission.xlsx), a Power BI dashboard was created for visualization.

Key Visuals:
Card: Total Claim Count.
Bar Chart: Claim count by Segment.
Column Chart: Claim count by Area Cluster.
Donut Chart: Claim count by Fuel Type.
Clustered Bar Chart: Average Claim Probability by NCAP Rating.

Data Model:
test.csv joined with submission.xlsx using policy_id.

CarInsurancePrediction/
│
├── app.py                     # Streamlit App
├── build_preprocessor.py      # Builds and saves preprocessing pipeline
├── retrain_model.py           # Retrains and saves LightGBM model
├── predict_test.py            # Generates predictions for test.csv
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
├── preprocessor.pkl           # Saved preprocessor
├── lgbm_best_model.pkl        # Trained model
├── submission.xlsx            # Final submission file
├── README.md                  # Project documentation (this file)
└── PowerBI_Dashboard.pbix     # Power BI file (if saved)

| Metric                  | Score    |
| ----------------------- | -------- |
| Validation AUC          | **0.63** |
| Accuracy                | **0.63** |
| Recall (Claim class)    | **0.54** |
| Precision (Claim class) | **0.09** |

Key Learnings

Handling class imbalance is critical for insurance data.
Feature engineering and categorical encoding impact model performance.
Power BI enhances business interpretability of ML results.
Streamlit provides user-friendly deployment for prediction systems.

👨‍💻 Author

Raji Ravi
Machine Learning Engineer & Data Science Enthusiast
📍 India

