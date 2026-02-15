import pandas as pd
import numpy as np

# ==============================
# STEP 1: LOAD DATA
# ==============================
df = pd.read_csv("apple_price_data.csv")

print("original data :\n",df)
print("="*100)
print(df.info())
print("="*100)
print(df.describe())
print("="*100)

# ==============================
# STEP 2: EDA + SCRUBBING
# ==============================
print("\n missing value:\n",df.isnull().sum())

df["date"] = pd.to_datetime(df["date"])

# forward fill missing
df.ffill(inplace=True)

# remove impossible values
df = df[df["price_today"] > 0]
df = df[df["arrival_qty"] >= 0]

# ==============================
# STEP 3: BINNING / BUCKETING
# ==============================
df["risk_ratio"] = df["days_since_harvest"] / df["max_safe_days"]

df["risk_bucket"] = pd.cut(
    df["risk_ratio"],
    bins=[0,0.5,0.8,1.5],
    labels=["low","medium","high"]
)

df["arrival_bucket"] = pd.cut(
    df["arrival_qty"],
    bins=[0,150,220,50000],
    labels=["LOW_SUPPLY","NORMAL","HIGH_SUPPLY"]
)

# ==============================
# STEP 4: FEATURE ENGINEERING
# ==============================
df["price_trend_1d"] = df["price_today"] - df["price_1d_ago"]
df["price_trend_3d"] = df["price_today"] - df["price_3d_ago"]

numeric_features = df[
    ["price_trend_1d","price_trend_3d","arrival_qty","risk_ratio"]
]

categorical = pd.get_dummies(
    df[["market","risk_bucket","arrival_bucket"]],
    drop_first=True
)

X = pd.concat([numeric_features,categorical],axis=1)
Y = df["price_tomorrow"]

print("="*100)
print("final ML ready data:\n",df)

# ==============================
# STEP 5: TRAIN TEST SPLIT
# ==============================
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)

# ==============================
# STEP 6: SCALING
# ==============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# STEP 7: MODEL SELECTION
# ==============================
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# ==============================
# STEP 8: MODEL TRAINING
# ==============================
model.fit(X_train,Y_train)

# ==============================
# STEP 9: PREDICTIONS
# ==============================
Y_pred = model.predict(X_test)

# ==============================
# STEP 10: EVALUATION
# ==============================
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

MSE = mean_squared_error(Y_test,Y_pred)
MAE = mean_absolute_error(Y_test,Y_pred)
R2  = r2_score(Y_test,Y_pred)

print(f"MSE:{MSE:.2f}")
print(f"MAE:{MAE:.2f}")
print(f"R2 Score:{R2:.2f}")
print("="*100)

# ==============================
# STEP 11: FULL DATA PREDICTION
# ==============================
X_scaled = scaler.transform(X)
df["predicted_price"] = model.predict(X_scaled)

# ==============================
# STEP 12: FARMER DECISION ENGINE
# ==============================
def farmer_decision(row):
    if row["risk_ratio"] >= 0.8:
        return "SELL (Crop Risk High)"
    elif row["predicted_price"] > row["price_today"]:
        return "WAIT (Price May Increase)"
    else:
        return "SELL (Price May Drop)"

df["decision"] = df.apply(farmer_decision,axis=1)

# ==============================
# STEP 13: BACKEND FUNCTION (FIXED)
# ==============================
def predict_crop_decision(input_df):

    input_df["risk_ratio"] = (
        input_df["days_since_harvest"] /
        input_df["max_safe_days"]
    )

    # --- FIX: Compute buckets (same as training) ---
    input_df["risk_bucket"] = pd.cut(
        input_df["risk_ratio"],
        bins=[0,0.5,0.8,1.5],
        labels=["low","medium","high"]
    )
    input_df["arrival_bucket"] = pd.cut(
        input_df["arrival_qty"],
        bins=[0,150,220,50000],
        labels=["LOW_SUPPLY","NORMAL","HIGH_SUPPLY"]
    )
    # ------------------------------------------------

    input_df["price_trend_1d"] = (
        input_df["price_today"] -
        input_df["price_1d_ago"]
    )
    input_df["price_trend_3d"] = (
        input_df["price_today"] -
        input_df["price_3d_ago"]
    )

    numeric = input_df[
        ["price_trend_1d","price_trend_3d",
         "arrival_qty","risk_ratio"]
    ]

    cat = pd.get_dummies(
        input_df[["market","risk_bucket","arrival_bucket"]],
        drop_first=True
    )

    X_input = pd.concat([numeric,cat],axis=1)

    # align with training columns
    X_input = X_input.reindex(columns=X.columns, fill_value=0)

    X_input_scaled = scaler.transform(X_input)

    pred = model.predict(X_input_scaled)

    input_df["predicted_price"] = pred

    return input_df.apply(farmer_decision,axis=1)

# ==============================
# STEP 14: OUTPUT FORMAT
# ==============================
print(
    df[
        ["date",
         "price_today",
         "predicted_price",
         "risk_bucket",
         "decision"]
    ].tail()
)

# ==============================
# STEP 15: VISUALISATION
# ==============================
from matplotlib import pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(Y_test, Y_pred)
plt.plot(
    [min(Y_test), max(Y_test)],
    [min(Y_test), max(Y_test)],
    linestyle='--'
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

print("="*100)
print("final data frame:\n",df)
print("="*100)

# ==============================
# TEST THE FIXED FUNCTION
# ==============================
print(" ========================== for testing ====================")
test_input = pd.DataFrame({
    "market":["Srinagar"],
    "arrival_qty":[8000],
    "price_today":[120],
    "price_1d_ago":[118],
    "price_3d_ago":[115],
    "days_since_harvest":[10],
    "max_safe_days":[180]
})

print(predict_crop_decision(test_input))