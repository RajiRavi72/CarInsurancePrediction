import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("📂 Loading dataset...")
data = pd.read_csv("car_insurance_data/train.csv")

# Drop ID and target columns
drop_cols = ["policy_id", "is_claim"]
for col in drop_cols:
    if col in data.columns:
        data = data.drop(columns=[col])

print("✅ Columns used for preprocessing:", list(data.columns))

# Define numeric and categorical columns
numeric_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = data.select_dtypes(include=["object"]).columns.tolist()

print("🔢 Numeric columns:", numeric_features)
print("🔤 Categorical columns:", categorical_features)

# Build preprocessor
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ✅ Fit the preprocessor
print("⚙️ Fitting preprocessor...")
preprocessor.fit(data)

# Save preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")
print("💾 Preprocessor saved as 'preprocessor.pkl'")
