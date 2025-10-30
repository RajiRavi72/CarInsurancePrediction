import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

print("📂 Loading training data...")
df = pd.read_csv("car_insurance_data/train.csv")

# Drop policy_id if it exists
if "policy_id" in df.columns:
    print("🧹 Dropping column: policy_id")
    df = df.drop(columns=["policy_id"])

X = df.drop(columns=["is_claim"])
y = df["is_claim"]

print("⚙️ Loading preprocessor...")
preprocessor = joblib.load("preprocessor.pkl")

print("🔄 Transforming features...")
X_transformed = preprocessor.transform(X)

# Split train-test for validation
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
pos_weight = (len(y) - sum(y)) / sum(y)
print(f"⚖️ Applying class weight scale_pos_weight={pos_weight:.2f}")

print("🚀 Training LightGBM model with class imbalance handling...")
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42
)

model.fit(X_train, y_train)

print("📊 Evaluating...")
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_proba)
print(f"✅ Validation AUC: {auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

print("💾 Saving model as 'lgbm_best_model.pkl'...")
joblib.dump(model, "lgbm_best_model.pkl")

print("🎉 Retraining complete! Try running the prediction script again.")
