# ============================================
# ECORIPE - ML POWERED APPLE PRICE PREDICTION
# UPDATED WITH INPUT VALIDATION & REALISTIC BOUNDS
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("="*60)
print("ECORIPE - LOADING APPLE PRICE DATA")
print("="*60)

df = pd.read_csv("apple_price_data.csv")

print(f"Data loaded: {len(df)} rows")
print(f"Markets: {df['market'].unique()}")

# ============================================
# STEP 2: EDA + SCRUBBING
# ============================================
df["date"] = pd.to_datetime(df["date"])
df.ffill(inplace=True)

# Remove impossible values
df = df[df["price_today"] > 0]
df = df[df["arrival_qty"] >= 0]
df = df[df["price_today"] < 500]  # Remove extreme outliers (>₹500)

print(f"Data cleaned: {len(df)} rows remaining")

# ============================================
# STEP 3: BINNING / BUCKETING
# ============================================
df["risk_ratio"] = df["days_since_harvest"] / df["max_safe_days"]

df["risk_bucket"] = pd.cut(
    df["risk_ratio"],
    bins=[0, 0.5, 0.8, 1.5],
    labels=["low", "medium", "high"]
)

df["arrival_bucket"] = pd.cut(
    df["arrival_qty"],
    bins=[0, 150, 220, 50000],
    labels=["LOW_SUPPLY", "NORMAL", "HIGH_SUPPLY"]
)

# ============================================
# STEP 4: FEATURE ENGINEERING
# ============================================
df["price_trend_1d"] = df["price_today"] - df["price_1d_ago"]
df["price_trend_3d"] = df["price_today"] - df["price_3d_ago"]

# Remove extreme trends (data errors)
df = df[df["price_trend_1d"].abs() < 50]  # Max ₹50 change in 1 day
df = df[df["price_trend_3d"].abs() < 100]  # Max ₹100 change in 3 days

numeric_features = df[
    ["price_trend_1d", "price_trend_3d", "arrival_qty", "risk_ratio"]
]

categorical = pd.get_dummies(
    df[["market", "risk_bucket", "arrival_bucket"]],
    drop_first=True
)

X = pd.concat([numeric_features, categorical], axis=1)
Y = df["price_tomorrow"]

print(f"Features created: {X.shape[1]} total features")

# ============================================
# STEP 5: TRAIN TEST SPLIT
# ============================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============================================
# STEP 6: SCALING
# ============================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# STEP 7: MODEL TRAINING
# ============================================
model = LinearRegression()
model.fit(X_train, Y_train)

# ============================================
# STEP 8: EVALUATION
# ============================================
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"\n MODEL PERFORMANCE:")
print(f"   R² Score: {r2:.3f}")
print(f"   MAE: ₹{mae:.2f}")
print(f"   MSE: {mse:.2f}")
print("="*60)

# ============================================
# STEP 9: FULL DATA PREDICTION
# ============================================
X_scaled = scaler.transform(X)
df["predicted_price"] = model.predict(X_scaled)

# ============================================
# STEP 10: FARMER DECISION ENGINE
# ============================================
def farmer_decision(row):
    if row["risk_ratio"] >= 0.8:
        return "SELL (Crop Risk High)"
    elif row["predicted_price"] > row["price_today"]:
        return "WAIT (Price May Increase)"
    else:
        return "SELL (Price May Drop)"

df["decision"] = df.apply(farmer_decision, axis=1)

# ============================================
# STEP 11: INPUT VALIDATION FUNCTION
# ============================================
def validate_input_data(input_df):
    """Validate and clip input data to realistic ranges"""
    warnings = []
    
    # Check price_today range
    if input_df["price_today"].iloc[0] < 20:
        input_df["price_today"] = 20
        warnings.append("Price too low, set to minimum ₹20")
    elif input_df["price_today"].iloc[0] > 300:
        input_df["price_today"] = 300
        warnings.append("Price too high, set to maximum ₹300")
    
    # Check price_1d_ago
    if input_df["price_1d_ago"].iloc[0] < 20:
        input_df["price_1d_ago"] = 20
    elif input_df["price_1d_ago"].iloc[0] > 300:
        input_df["price_1d_ago"] = 300
    
    # Check price_3d_ago
    if input_df["price_3d_ago"].iloc[0] < 20:
        input_df["price_3d_ago"] = 20
    elif input_df["price_3d_ago"].iloc[0] > 300:
        input_df["price_3d_ago"] = 300
    
    # Check for unrealistic 1-day change
    price_change_1d = input_df["price_today"].iloc[0] - input_df["price_1d_ago"].iloc[0]
    if abs(price_change_1d) > 40:
        input_df["price_1d_ago"] = input_df["price_today"].iloc[0] - (40 if price_change_1d > 0 else -40)
        warnings.append("1-day price change capped at ₹40 (realistic limit)")
    
    # Check arrival_qty
    if input_df["arrival_qty"].iloc[0] < 100:
        input_df["arrival_qty"] = 100
        warnings.append("Arrival quantity too low, set to minimum 100 kg")
    elif input_df["arrival_qty"].iloc[0] > 50000:
        input_df["arrival_qty"] = 50000
        warnings.append("Arrival quantity too high, set to maximum 50,000 kg")
    
    # Check days_since_harvest
    if input_df["days_since_harvest"].iloc[0] < 0:
        input_df["days_since_harvest"] = 0
    elif input_df["days_since_harvest"].iloc[0] > 365:
        input_df["days_since_harvest"] = 365
    
    # Check max_safe_days
    if input_df["max_safe_days"].iloc[0] < 1:
        input_df["max_safe_days"] = 1
    elif input_df["max_safe_days"].iloc[0] > 365:
        input_df["max_safe_days"] = 365
    
    return input_df, warnings

# ============================================
# STEP 12: POST-PROCESSING (REALISTIC PREDICTIONS)
# ============================================
def post_process_prediction(predicted_price, input_df):
    """Ensure predictions stay within realistic bounds"""
    
    price_today = input_df["price_today"].iloc[0]
    
    # Predictions shouldn't change more than ±30% in a day
    max_increase = price_today * 1.30
    max_decrease = price_today * 0.70
    
    if predicted_price > max_increase:
        predicted_price = max_increase
    elif predicted_price < max_decrease:
        predicted_price = max_decrease
    
    # Overall price bounds
    if predicted_price < 20:
        predicted_price = 20
    elif predicted_price > 300:
        predicted_price = 300
    
    return predicted_price

# ============================================
# STEP 13: ENHANCED PREDICTION FUNCTION
# ============================================
def predict_crop_decision_enhanced(input_df):
    """Enhanced prediction with validation and realistic bounds"""
    
    # Step 1: Validate input
    input_df, warnings = validate_input_data(input_df)
    
    # Step 2: Calculate risk ratio
    input_df["risk_ratio"] = (
        input_df["days_since_harvest"] / 
        input_df["max_safe_days"]
    )
    
    # Step 3: Create buckets
    input_df["risk_bucket"] = pd.cut(
        input_df["risk_ratio"],
        bins=[0, 0.5, 0.8, 1.5],
        labels=["low", "medium", "high"]
    )
    input_df["arrival_bucket"] = pd.cut(
        input_df["arrival_qty"],
        bins=[0, 150, 220, 50000],
        labels=["LOW_SUPPLY", "NORMAL", "HIGH_SUPPLY"]
    )
    
    # Step 4: Calculate price trends
    input_df["price_trend_1d"] = (
        input_df["price_today"] - input_df["price_1d_ago"]
    )
    input_df["price_trend_3d"] = (
        input_df["price_today"] - input_df["price_3d_ago"]
    )
    
    # Step 5: Prepare features
    numeric = input_df[
        ["price_trend_1d", "price_trend_3d",
         "arrival_qty", "risk_ratio"]
    ]
    
    cat = pd.get_dummies(
        input_df[["market", "risk_bucket", "arrival_bucket"]],
        drop_first=True
    )
    
    X_input = pd.concat([numeric, cat], axis=1)
    X_input = X_input.reindex(columns=X.columns, fill_value=0)
    
    # Step 6: Scale and predict
    X_input_scaled = scaler.transform(X_input)
    raw_prediction = model.predict(X_input_scaled)[0]
    
    # Step 7: Post-process for realistic values
    final_prediction = post_process_prediction(raw_prediction, input_df)
    input_df["predicted_price"] = final_prediction
    
    # Step 8: Get decision
    decision = input_df.apply(farmer_decision, axis=1).iloc[0]
    
    return {
        "decision": decision,
        "predicted_price": round(final_prediction, 2),
        "risk_ratio": round(input_df["risk_ratio"].iloc[0], 3),
        "price_trend_1d": round(input_df["price_trend_1d"].iloc[0], 2),
        "price_trend_3d": round(input_df["price_trend_3d"].iloc[0], 2),
        "warnings": warnings,
        "confidence": round(r2 * 100, 1)
    }

# ============================================
# STEP 14: BATCH PREDICTION FOR MULTIPLE MARKETS
# ============================================
def predict_multiple_markets(markets_data):
    """Predict for multiple markets at once"""
    results = []
    for market_data in markets_data:
        input_df = pd.DataFrame([market_data])
        result = predict_crop_decision_enhanced(input_df)
        result.update(market_data)
        results.append(result)
    return results

# ============================================
# STEP 15: TEST THE ENHANCED FUNCTION
# ============================================
print("\n" + "="*60)
print("TESTING ECORIPE PREDICTION ENGINE")
print("="*60)

# Test Case 1: Normal Srinagar data
test1 = pd.DataFrame({
    "market": ["Srinagar"],
    "arrival_qty": [8000],
    "price_today": [120],
    "price_1d_ago": [118],
    "price_3d_ago": [115],
    "days_since_harvest": [10],
    "max_safe_days": [180]
})

result1 = predict_crop_decision_enhanced(test1)
print(f"\nTest 1 - Normal Data:")
print(f"   Decision: {result1['decision']}")
print(f"   Predicted: ₹{result1['predicted_price']}")
print(f"   Risk Ratio: {result1['risk_ratio']}")
print(f"   Confidence: {result1['confidence']}%")

# Test Case 2: Extreme data (should be corrected)
test2 = pd.DataFrame({
    "market": ["Srinagar"],
    "arrival_qty": [8000],
    "price_today": [120],
    "price_1d_ago": [60],  # Unrealistic!
    "price_3d_ago": [100],
    "days_since_harvest": [10],
    "max_safe_days": [180]
})

result2 = predict_crop_decision_enhanced(test2)
print(f"\nTest 2 - Extreme Data (with validation):")
print(f"   Decision: {result2['decision']}")
print(f"   Predicted: ₹{result2['predicted_price']}")
print(f"   Risk Ratio: {result2['risk_ratio']}")
if result2['warnings']:
    print(f"   Warnings: {result2['warnings']}")

# Test Case 3: Different market
test3 = pd.DataFrame({
    "market": ["Delhi"],
    "arrival_qty": [5000],
    "price_today": [95],
    "price_1d_ago": [93],
    "price_3d_ago": [90],
    "days_since_harvest": [5],
    "max_safe_days": [150]
})

result3 = predict_crop_decision_enhanced(test3)
print(f"\nTest 3 - Delhi Market:")
print(f"   Decision: {result3['decision']}")
print(f"   Predicted: ₹{result3['predicted_price']}")
print(f"   Risk Ratio: {result3['risk_ratio']}")

print("\n" + "="*60)
print("ECORIPE BACKEND READY FOR PRODUCTION!")
print("="*60)

# ============================================
# STEP 16: EXPORT FOR API USE
# ============================================
# These will be used by auth_backend.py
__all__ = ['predict_crop_decision_enhanced', 'model', 'scaler', 'X', 'r2']
