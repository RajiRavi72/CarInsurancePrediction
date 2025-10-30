# generate_submission.py

import pandas as pd
import joblib

# 1️⃣ Load the test data
print("📂 Loading test data...")
test = pd.read_csv("car_insurance_data/test.csv")

# 2️⃣ Load preprocessor and trained model
print("⚙️ Loading preprocessor and model...")
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("lgbm_best_model.pkl")

# 3️⃣ Preserve policy_id
policy_ids = test["policy_id"]

# 4️⃣ Drop policy_id before transforming
if "policy_id" in test.columns:
    test = test.drop(columns=["policy_id"])

# 5️⃣ Transform features
print("🔄 Transforming test data...")
X_test = preprocessor.transform(test)

# 6️⃣ Predict probabilities and class labels
print("🤖 Generating predictions...")
y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of claim = 1
y_pred_class = (y_pred_prob >= 0.5).astype(int)  # threshold 0.5

# 7️⃣ Create submission DataFrame
submission = pd.DataFrame({
    "policy_id": policy_ids,
    "is_claim": y_pred_class
})

# 8️⃣ Save to CSV/Excel
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")

# Optional: also create Excel version
submission.to_excel("submission.xlsx", index=False)
print("✅ submission.xlsx created successfully!")
