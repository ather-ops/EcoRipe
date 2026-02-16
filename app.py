# ============================================
# ECORIPE - APPLE PRICE PREDICTION ENGINE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# STEP 1: LOAD YOUR DATA
# ============================================
# This is your training data - the brain of your model
print("="*60)
print(" ECORIPE - LOADING APPLE DATA")
print("="*60)

# Read the CSV file
df = pd.read_csv("apple_price_data.csv")

# See what's inside
print(f" Loaded {len(df)} rows of data")
print(f" Markets: {df['market'].unique()}")
print(f" Columns: {list(df.columns)}")

# ============================================
# STEP 2: CLEAN YOUR DATA
# ============================================
# Remove bad data that would confuse the model

# Convert dates to proper format
df["date"] = pd.to_datetime(df["date"])

# Fill missing values (forward fill means use previous value)
df.ffill(inplace=True)

# Remove impossible prices (can't have negative or zero)
df = df[df["price_today"] > 0]
df = df[df["price_tomorrow"] > 0]  # Can't predict negative prices

# Remove crazy outliers (prices above ₹500 are unrealistic)
df = df[df["price_today"] < 500]
df = df[df["price_tomorrow"] < 500]

print(f"After cleaning: {len(df)} rows remain")

# ============================================
# STEP 3: CREATE SMART FEATURES
# ============================================
# These are the patterns your model will learn

# RISK RATIO: How old is the apple compared to its max life?
# If days_since_harvest = 100, max_safe_days = 200
# Then risk_ratio = 0.5 (medium risk)
df["risk_ratio"] = df["days_since_harvest"] / df["max_safe_days"]

# PRICE TRENDS: Is price going up or down?
# If today's price is higher than yesterday, trend is positive
df["price_trend_1d"] = df["price_today"] - df["price_1d_ago"]
df["price_trend_3d"] = df["price_today"] - df["price_3d_ago"]

# Remove extreme trends (data errors)
# If price changed by more than ₹50 in one day, something's wrong
df = df[df["price_trend_1d"].abs() < 50]
df = df[df["price_trend_3d"].abs() < 100]

# ============================================
# STEP 4: CATEGORIZE THINGS (BINNING)
# ============================================
# Turn numbers into categories so model understands them better

# RISK BUCKETS: Low, Medium, High
# 0 to 0.5 = Low risk (fresh apples)
# 0.5 to 0.8 = Medium risk (getting older)
# 0.8 to 1.5 = High risk (must sell soon!)
df["risk_bucket"] = pd.cut(
    df["risk_ratio"],
    bins=[0, 0.5, 0.8, 1.5],
    labels=["low", "medium", "high"]
)

# SUPPLY BUCKETS: Low, Normal, High supply
# 0 to 150 kg = Low supply (rare, prices may rise)
# 150 to 220 kg = Normal supply
# 220+ kg = High supply (plenty, prices may drop)
df["arrival_bucket"] = pd.cut(
    df["arrival_qty"],
    bins=[0, 150, 220, 50000],
    labels=["LOW_SUPPLY", "NORMAL", "HIGH_SUPPLY"]
)

# ============================================
# STEP 5: PREPARE DATA FOR MODEL
# ============================================
# Split into numbers and categories

# Numeric features (the actual numbers)
numeric_features = df[
    ["price_trend_1d",    # Is price going up/down?
     "price_trend_3d",    # Same but over 3 days
     "arrival_qty",       # How many apples arrived?
     "risk_ratio"]        # How risky is this crop?
]

# Categorical features (convert to 0s and 1s)
# This is called "one-hot encoding"
# Example: market=Srinagar becomes [1,0,0] etc.
categorical = pd.get_dummies(
    df[["market", "risk_bucket", "arrival_bucket"]],
    drop_first=True  # Avoid duplicate info
)

# Combine everything into one big table
X = pd.concat([numeric_features, categorical], axis=1)

# Y is what we want to predict (tomorrow's price)
Y = df["price_tomorrow"]

print(f"Created {X.shape[1]} features for training")

# ============================================
# STEP 6: SPLIT DATA FOR TRAINING & TESTING
# ============================================
# Train on 80% of data, test on 20%
# This tells us if model actually learned or just memorized

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============================================
# STEP 7: SCALE THE NUMBERS
# ============================================
# Make all features comparable (no feature dominates just because it's bigger)
# Example: arrival_qty (8000) vs risk_ratio (0.5)
# Scaling puts them both in similar ranges

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Learn the scaling from training data
X_test = scaler.transform(X_test)        # Apply same scaling to test data

# ============================================
# STEP 8: TRAIN THE MODEL
# ============================================
# This is where the magic happens!
# Linear Regression finds the best formula: 
# price = (weight1 × feature1) + (weight2 × feature2) + ... + constant

model = LinearRegression()
model.fit(X_train, Y_train)

# ============================================
# STEP 9: CHECK HOW GOOD THE MODEL IS
# ============================================
# Make predictions on test data (data model hasn't seen)

Y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(Y_test, Y_pred)        # Average squared error
mae = mean_absolute_error(Y_test, Y_pred)       # Average error in rupees
r2 = r2_score(Y_test, Y_pred)                   # How well model explains price changes

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   R² Score: {r2:.3f} (1.0 is perfect, 0 is useless)")
print(f"   Average Error: ₹{mae:.2f}")
print(f"   This means predictions are usually off by about ₹{mae:.2f}")

# ============================================
# STEP 10: ADD REALISTIC BOUNDS (IMPORTANT!)
# ============================================
# This prevents the model from making crazy predictions

def apply_realistic_bounds(predicted_price, price_today, price_1d_ago, price_3d_ago):
    """
    Make sure predictions are realistic for fruit markets
    """
    
    # 1. Calculate average price from recent days
    avg_recent = (price_today + price_1d_ago + price_3d_ago) / 3
    
    # 2. Price can't be less than 70% of recent average (too low)
    #    or more than 130% of recent average (too high)
    min_allowed = avg_recent * 0.70
    max_allowed = avg_recent * 1.30
    
    # 3. Apply the bounds
    if predicted_price < min_allowed:
        return min_allowed
    elif predicted_price > max_allowed:
        return max_allowed
    else:
        return predicted_price

# ============================================
# STEP 11: FARMER DECISION LOGIC (YOUR BRAIN!)
# ============================================
# This is YOUR logic. You wrote this. Own it.

def farmer_decision(risk_ratio, price_today, predicted_price):
    """
    Tell farmer what to do based on risk and price
    """
    if risk_ratio >= 0.8:
        return "SELL (Crop Risk High)"
    elif predicted_price > price_today:
        return "WAIT (Price May Increase)"
    else:
        return "SELL (Price May Drop)"

# ============================================
# STEP 12: MAIN PREDICTION FUNCTION
# ============================================
# This is what the API calls. YOU control this.

def predict_crop_decision(input_df):
    """
    Takes farmer's input, returns prediction
    
    Input should have:
    - market: "Srinagar", "Delhi", etc.
    - arrival_qty: how many kg arrived
    - price_today: current price
    - price_1d_ago: yesterday's price
    - price_3d_ago: price from 3 days ago
    - days_since_harvest: how old the crop is
    - max_safe_days: how long it can store
    """
    
    # ========================================
    # STEP 12a: Calculate risk ratio
    # ========================================
    input_df["risk_ratio"] = (
        input_df["days_since_harvest"] / 
        input_df["max_safe_days"]
    )
    
    # ========================================
    # STEP 12b: Create buckets (same as training)
    # ========================================
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
    
    # ========================================
    # STEP 12c: Calculate price trends
    # ========================================
    input_df["price_trend_1d"] = (
        input_df["price_today"] - input_df["price_1d_ago"]
    )
    input_df["price_trend_3d"] = (
        input_df["price_today"] - input_df["price_3d_ago"]
    )
    
    # ========================================
    # STEP 12d: Prepare features for model
    # ========================================
    numeric = input_df[
        ["price_trend_1d", "price_trend_3d",
         "arrival_qty", "risk_ratio"]
    ]
    
    cat = pd.get_dummies(
        input_df[["market", "risk_bucket", "arrival_bucket"]],
        drop_first=True
    )
    
    # Combine numbers and categories
    X_input = pd.concat([numeric, cat], axis=1)
    
    # Make sure columns match training data
    # (if a category is missing, fill with 0)
    X_input = X_input.reindex(columns=X.columns, fill_value=0)
    
    # ========================================
    # STEP 12e: Scale and predict
    # ========================================
    X_input_scaled = scaler.transform(X_input)
    raw_prediction = model.predict(X_input_scaled)[0]
    
    # ========================================
    # STEP 12f: Apply realistic bounds
    # ========================================
    price_today = input_df["price_today"].iloc[0]
    price_1d_ago = input_df["price_1d_ago"].iloc[0]
    price_3d_ago = input_df["price_3d_ago"].iloc[0]
    
    final_prediction = apply_realistic_bounds(
        raw_prediction, 
        price_today, 
        price_1d_ago, 
        price_3d_ago
    )
    
    input_df["predicted_price"] = final_prediction
    
    # ========================================
    # STEP 12g: Make decision for farmer
    # ========================================
    risk_ratio = input_df["risk_ratio"].iloc[0]
    decision = farmer_decision(risk_ratio, price_today, final_prediction)
    
    # ========================================
    # STEP 12h: Return everything
    # ========================================
    return {
        "decision": decision,
        "predicted_price": round(final_prediction, 2),
        "risk_ratio": round(risk_ratio, 3),
        "price_trend_1d": round(input_df["price_trend_1d"].iloc[0], 2),
        "price_trend_3d": round(input_df["price_trend_3d"].iloc[0], 2),
        "confidence": round(r2 * 100, 1),  # How much to trust this
        "note": "Prediction bounded to ±30% of recent average for realism"
    }

# ============================================
# STEP 13: TEST YOUR FUNCTION
# ============================================
print("\n" + "="*60)
print("TESTING YOUR PREDICTION ENGINE")
print("="*60)

# Test Case 1: Normal Srinagar data (should give reasonable result)
test1 = pd.DataFrame({
    "market": ["Srinagar"],
    "arrival_qty": [8000],
    "price_today": [65],
    "price_1d_ago": [56],
    "price_3d_ago": [70],
    "days_since_harvest": [100],
    "max_safe_days": [180]
})

result1 = predict_crop_decision(test1)
print(f"\n TEST 1 - Srinagar Market:")
print(f"   Input: Today ₹65, Yesterday ₹56, 3 days ago ₹70")
print(f"   Output: Predicted ₹{result1['predicted_price']}")
print(f"   Decision: {result1['decision']}")
print(f"   Risk Ratio: {result1['risk_ratio']}")
print(f"   Confidence: {result1['confidence']}%")

# Test Case 2: Extreme data (should be capped)
test2 = pd.DataFrame({
    "market": ["Srinagar"],
    "arrival_qty": [8000],
    "price_today": [65],
    "price_1d_ago": [45],  # Unrealistic jump!
    "price_3d_ago": [70],
    "days_since_harvest": [100],
    "max_safe_days": [180]
})

result2 = predict_crop_decision(test2)
print(f"\n TEST 2 - Extreme Data (capped for realism):")
print(f"   Input: Today ₹65, Yesterday ₹45 (unrealistic)")
print(f"   Raw prediction would be crazy, but bounded version:")
print(f"   Output: ₹{result2['predicted_price']} (much more realistic!)")

# ============================================
# STEP 14: EXPORT FOR API USE
# ============================================
# These are the things auth_backend.py will import
__all__ = ['predict_crop_decision', 'model', 'scaler', 'X', 'r2']

print("\n" + "="*60)
print("YOUR BACKEND IS READY!")
print("="*60)
print("\n Exported for API:")
print("   - predict_crop_decision()  → Main prediction function")
print("   - model                     → Trained ML model")
print("   - scaler                    → For scaling inputs")
print("   - X                         → Feature names")
print("   - r2                        → Model accuracy")
