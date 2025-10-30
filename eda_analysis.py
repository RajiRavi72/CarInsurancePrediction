# eda_analysis.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Load the data
base_path = "car_insurance_data"
train_df = pd.read_csv(os.path.join(base_path, "train.csv"))

print("✅ Training data loaded for EDA")
print("Shape:", train_df.shape)

# Target variable
target_col = "is_claim"

# 1️⃣ Target Distribution
fig = px.histogram(train_df, x=target_col, color=target_col,
                   title="Distribution of Target Variable: is_claim",
                   text_auto=True)
fig.show()

# 2️⃣ Policy Tenure vs Claim
fig = px.box(train_df, x=target_col, y="policy_tenure",
             title="Policy Tenure vs Claim")
fig.show()

# 3️⃣ Vehicle Age vs Claim
fig = px.box(train_df, x=target_col, y="age_of_car",
             title="Vehicle Age vs Claim")
fig.show()

# 4️⃣ Policyholder Age vs Claim
fig = px.box(train_df, x=target_col, y="age_of_policyholder",
             title="Policyholder Age vs Claim")
fig.show()

# 5️⃣ Claims by Vehicle Segment
fig = px.histogram(train_df, x="segment", color=target_col, barmode="group",
                   title="Claims by Vehicle Segment")
fig.show()

# 6️⃣ Claims by Fuel Type
fig = px.histogram(train_df, x="fuel_type", color=target_col, barmode="group",
                   title="Claims by Fuel Type")
fig.show()

# 7️⃣ Claims by Transmission Type
fig = px.histogram(train_df, x="transmission_type", color=target_col, barmode="group",
                   title="Claims by Transmission Type")
fig.show()

# 8️⃣ Correlation Heatmap (Numeric Features)
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
corr = train_df[num_cols].corr()
fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap (Numeric Features)")
fig.show()

# 9️⃣ Safety Features vs Claim (Example)
safety_features = [
    "is_esc", "is_brake_assist", "is_power_steering",
    "is_parking_camera", "is_parking_sensors", "is_ecw"
]

for feature in safety_features:
    if feature in train_df.columns:
        fig = px.histogram(train_df, x=feature, color=target_col, barmode="group",
                           title=f"{feature} vs Claim")
        fig.show()

print("\n📊 Interactive visualizations displayed successfully!")
