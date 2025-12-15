import sys
import json
import pandas as pd
import numpy as np
import warnings
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Attempt to import XGBoost as per notebook methodology [cite: 19]
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not found. Performance will be degraded (fallback to GBT only).")

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
TRAIN_FILE = "public_cases.json"
MODEL_BUNDLE_FILE = "model_bundle.joblib"
RANDOM_STATE = 42

# ==========================================
# 1. CONSTANTS (Tuned from Notebook)
# Source: [cite: 49-84]
# ==========================================
PER_DIEM_RATE = 100.0
MILEAGE_TIER1_THRESHOLD = 100
MILEAGE_TIER2_THRESHOLD = 300
MILEAGE_RATE_TIER1 = 0.58
MILEAGE_RATE_TIER2 = 0.21  # Tuned from original 0.30
MILEAGE_RATE_TIER3 = 0.28  # Tuned from original 0.22

RECEIPT_LOW_THRESHOLD = 50.0
RECEIPT_LOW_PENALTY_FACTOR = -0.10
RECEIPT_SWEET_SPOT_START = 600.0
RECEIPT_SWEET_SPOT_END = 800.0
RECEIPT_SWEET_SPOT_RATE = 0.70 # Tuned from 0.90
RECEIPT_DEFAULT_RATE = 0.15    # Tuned from 0.50
RECEIPT_HIGH_AMOUNT_RATE = 0.12

FIVE_DAY_TRIP_BONUS = 150.0
EFFICIENCY_BONUS_AMOUNT = 50.0
EFFICIENCY_MILES_MIN = 180.0
EFFICIENCY_MILES_MAX = 220.0
EFFICIENCY_RECEIPTS_MAX_SHORT = 75.0
EFFICIENCY_RECEIPTS_MAX_MED = 120.0
EFFICIENCY_RECEIPTS_MAX_LONG = 90.0

LOW_SPEND_PENALTY_AMOUNT = 75.0
LOW_SPEND_RECEIPTS_PER_DAY_THRESHOLD = 20.0

LONG_TRIP_PENALTY_AMOUNT = 100.0
LONG_TRIP_DAYS_THRESHOLD = 8
LONG_TRIP_RECEIPT_THRESHOLD = 90.0

ROUNDING_BUG_CENTS = {49, 99}
ROUNDING_BUG_AMOUNT = 0.50

# ==========================================
# 2. ANCHOR MODEL (Baseline Logic)
# Source: [cite: 89-145]
# ==========================================
def calculate_reimbursement(days: int, miles: float, receipts: float):
    # Per Diem
    total = days * PER_DIEM_RATE
    
    # Mileage (3-tier)
    rem = miles
    m1 = min(rem, MILEAGE_TIER1_THRESHOLD)
    total += m1 * MILEAGE_RATE_TIER1
    rem -= m1
    if rem > 0:
        m2 = min(rem, MILEAGE_TIER2_THRESHOLD - MILEAGE_TIER1_THRESHOLD)
        total += m2 * MILEAGE_RATE_TIER2
        rem -= m2
    if rem > 0:
        total += rem * MILEAGE_RATE_TIER3
        
    # Receipts (4-tier)
    if receipts < RECEIPT_LOW_THRESHOLD and days > 1:
        total += receipts * RECEIPT_LOW_PENALTY_FACTOR
    elif RECEIPT_SWEET_SPOT_START <= receipts <= RECEIPT_SWEET_SPOT_END:
        total += receipts * RECEIPT_SWEET_SPOT_RATE
    elif receipts > RECEIPT_SWEET_SPOT_END:
        total += (RECEIPT_SWEET_SPOT_END * RECEIPT_SWEET_SPOT_RATE) + \
                 ((receipts - RECEIPT_SWEET_SPOT_END) * RECEIPT_HIGH_AMOUNT_RATE)
    else:
        total += receipts * RECEIPT_DEFAULT_RATE
        
    # Bonuses / Penalties
    mpd = miles / days if days > 0 else 0.0
    rpd = receipts / days if days > 0 else 0.0
    
    # Lisa's 5-Day Bonus
    if days == 5:
        total += FIVE_DAY_TRIP_BONUS
        
    # Kevin's Efficiency Bonus
    is_efficient_miles = (EFFICIENCY_MILES_MIN <= mpd <= EFFICIENCY_MILES_MAX)
    is_modest_spending = False
    
    if days < 4:
        is_modest_spending = (rpd < EFFICIENCY_RECEIPTS_MAX_SHORT)
    elif 4 <= days <= 6:
        is_modest_spending = (rpd < EFFICIENCY_RECEIPTS_MAX_MED)
    else:
        is_modest_spending = (rpd < EFFICIENCY_RECEIPTS_MAX_LONG)
        
    if is_efficient_miles and is_modest_spending:
        total += EFFICIENCY_BONUS_AMOUNT
        
    # Low Spend Penalty
    if rpd < LOW_SPEND_RECEIPTS_PER_DAY_THRESHOLD and days > 1:
        total -= LOW_SPEND_PENALTY_AMOUNT
        
    # Vacation Penalty
    if days >= LONG_TRIP_DAYS_THRESHOLD and rpd > LONG_TRIP_RECEIPT_THRESHOLD:
        total -= LONG_TRIP_PENALTY_AMOUNT
        
    # Rounding Quirk
    cents = int(round(receipts * 100)) % 100
    if cents in ROUNDING_BUG_CENTS:
        total += ROUNDING_BUG_AMOUNT
        
    return round(float(total), 2)

# ==========================================
# 3. FEATURE ENGINEERING
# Source: [cite: 286-317]
# ==========================================
def add_residual_features(frame):
    f = frame.copy()
    
    # Calculate Baseline for Residual
    f['baseline_pred'] = f.apply(lambda r: calculate_reimbursement(
        r['trip_duration_days'], r['miles_traveled'], r['total_receipts_amount']), axis=1)

    # Ratios & Transforms
    f["miles_per_day"] = f["miles_traveled"] / (f["trip_duration_days"] + 1e-6)
    f["receipts_per_day"] = f["total_receipts_amount"] / (f["trip_duration_days"] + 1e-6)
    f["log_receipts"] = np.log1p(f["total_receipts_amount"])
    f["sqrt_miles"] = np.sqrt(np.maximum(f["miles_traveled"], 0))
    
    # Flags derived from Rules
    f["is_5day"] = (f["trip_duration_days"] == 5).astype(int)
    f["is_long_trip"] = (f["trip_duration_days"] >= LONG_TRIP_DAYS_THRESHOLD).astype(int)
    f["is_sweet_spot"] = ((f["total_receipts_amount"] >= RECEIPT_SWEET_SPOT_START) & 
                          (f["total_receipts_amount"] <= RECEIPT_SWEET_SPOT_END)).astype(int)
    f["is_receipts_over_800"] = (f["total_receipts_amount"] > RECEIPT_SWEET_SPOT_END).astype(int)
    f["is_receipts_under_50"] = (f["total_receipts_amount"] < RECEIPT_LOW_THRESHOLD).astype(int)
    f["is_rpd_under_20"] = (f["receipts_per_day"] < LOW_SPEND_RECEIPTS_PER_DAY_THRESHOLD).astype(int)
    
    f["is_efficient_miles"] = ((f["miles_per_day"] >= EFFICIENCY_MILES_MIN) & 
                               (f["miles_per_day"] <= EFFICIENCY_MILES_MAX)).astype(int)
    
    f["is_modest_short"] = ((f["trip_duration_days"] < 4) & (f["receipts_per_day"] < EFFICIENCY_RECEIPTS_MAX_SHORT)).astype(int)
    f["is_modest_med"] = ((f["trip_duration_days"] >= 4) & (f["trip_duration_days"] <= 6) & (f["receipts_per_day"] < EFFICIENCY_RECEIPTS_MAX_MED)).astype(int)
    f["is_modest_long"] = ((f["trip_duration_days"] > 6) & (f["receipts_per_day"] < EFFICIENCY_RECEIPTS_MAX_LONG)).astype(int)

    f["is_vacation_penalty"] = ((f["trip_duration_days"] >= LONG_TRIP_DAYS_THRESHOLD) & 
                                (f["receipts_per_day"] > LONG_TRIP_RECEIPT_THRESHOLD)).astype(int)
    
    f["is_miles_tier1"] = (f["miles_traveled"] < MILEAGE_TIER1_THRESHOLD).astype(int)
    f["is_miles_tier2"] = ((f["miles_traveled"] >= MILEAGE_TIER1_THRESHOLD) & 
                           (f["miles_traveled"] < MILEAGE_TIER2_THRESHOLD)).astype(int)
    
    # Cents Quirks
    cents = (np.round(f["total_receipts_amount"] * 100) % 100).astype(int)
    f["is_cents_49"] = (cents == 49).astype(int)
    f["is_cents_99"] = (cents == 99).astype(int)
    
    # Interactions
    f["days_x_miles"] = f["trip_duration_days"] * f["miles_traveled"]
    f["days_x_receipts"] = f["trip_duration_days"] * f["total_receipts_amount"]
    f["miles_x_receipts"] = f["miles_traveled"] * f["total_receipts_amount"]
    f["miles_sq"] = f["miles_traveled"]**2
    f["receipts_sq"] = f["total_receipts_amount"]**2
    
    return f

# Exact Feature List from Source 320
FEATS = [
    "trip_duration_days", "miles_traveled", "total_receipts_amount",
    "miles_per_day", "receipts_per_day", "log_receipts", "sqrt_miles",
    "is_5day", "is_long_trip", "is_sweet_spot", "is_receipts_over_800",
    "is_receipts_under_50", "is_rpd_under_20", "is_efficient_miles",
    "is_modest_short", "is_modest_med", "is_modest_long", "is_vacation_penalty",
    "is_miles_tier1", "is_miles_tier2", "is_cents_49", "is_cents_99",
    "days_x_miles", "days_x_receipts", "miles_x_receipts",
    "miles_sq", "receipts_sq"
]

# ==========================================
# 4. TRAINING & LOADING LOGIC
# Source: [cite: 331-413]
# ==========================================
def load_data(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        rows = []
        for entry in data:
            row = entry["input"]
            row["expected_output"] = entry["expected_output"]
            rows.append(row)
        return pd.DataFrame(rows)
    except FileNotFoundError:
        return None

def train_and_get_models():
    """
    Trains the XGBoost + GBT Ensemble and the Isotonic Calibrator.
    Returns a dictionary of models.
    """
    if os.path.exists(MODEL_BUNDLE_FILE):
        return joblib.load(MODEL_BUNDLE_FILE)

    print("Training new models from scratch...")
    df = load_data(TRAIN_FILE)
    if df is None:
        raise FileNotFoundError(f"Could not find {TRAIN_FILE} to train models.")

    # 1. Prepare Features & Residuals
    df = add_residual_features(df)
    df["residual"] = df["expected_output"] - df["baseline_pred"]
    
    X_all = df[FEATS].astype(float).values
    y_all = df["residual"].values
    
    # Split for Early Stopping validation
    Xtr, Xva, ytr, yva = train_test_split(X_all, y_all, test_size=0.25, random_state=RANDOM_STATE)
    
    # 2. Train XGBoost (Model A) [cite: 340-354]
    xgb = None
    if XGB_AVAILABLE:
        print("Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=2000, max_depth=4, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=RANDOM_STATE, tree_method="hist",
            early_stopping_rounds=50
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    
    # 3. Train GradientBoosting (Model B) [cite: 360]
    print("Training GradientBoosting...")
    gbt = GradientBoostingRegressor(
        n_estimators=800, max_depth=4, learning_rate=0.03,
        loss='absolute_error', random_state=RANDOM_STATE
    )
    gbt.fit(Xtr, ytr)
    # Note: Staged predict optimization omitted for brevity in production script,
    # relying on robust defaults or pre-calculated optima.
    
    # 4. Train Isotonic Calibrator [cite: 466]
    # We generate "uncalibrated" predictions on the FULL dataset to learn the bias map.
    print("Calibrating...")
    pred_gbt_full = gbt.predict(X_all)
    if xgb:
        pred_xgb_full = xgb.predict(X_all)
        blended_resid = (pred_gbt_full * 0.5) + (pred_xgb_full * 0.5)
    else:
        blended_resid = pred_gbt_full
        
    final_pred_uncalibrated = df["baseline_pred"].values + blended_resid
    
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(final_pred_uncalibrated, df["expected_output"].values)
    
    # Bundle and Save
    bundle = {
        "xgb": xgb,
        "gbt": gbt,
        "iso": iso,
        "features": FEATS
    }
    joblib.dump(bundle, MODEL_BUNDLE_FILE)
    print("Models saved.")
    return bundle

# ==========================================
# 5. INFERENCE LOGIC
# Source: [cite: 414-431]
# ==========================================
def main():
    if len(sys.argv) != 4:
        print("Usage: python reimbursement_calculator.py <days> <miles> <receipts>")
        return

    try:
        duration = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
    except ValueError:
        print("0.00")
        return

    # Load Models
    models = train_and_get_models()
    
    # 1. Baseline
    base_pred = calculate_reimbursement(duration, miles, receipts)
    
    # 2. Features
    # Note: We must create a DataFrame to match the training pipeline structure
    row = pd.DataFrame([{
        "trip_duration_days": duration,
        "miles_traveled": miles,
        "total_receipts_amount": receipts
    }])
    row = add_residual_features(row)
    X = row[models["features"]].astype(float).values
    
    # 3. Blended Residual
    pred_gbt = models["gbt"].predict(X)[0]
    
    if models["xgb"]:
        pred_xgb = models["xgb"].predict(X)[0]
        # Blend: 50% XGB + 50% GBT [cite: 426]
        resid = (pred_gbt * 0.5) + (pred_xgb * 0.5)
    else:
        resid = pred_gbt
        
    # 4. Uncalibrated Total
    uncalibrated_total = base_pred + resid
    
    # 5. Calibration [cite: 473]
    final_calibrated = models["iso"].predict([uncalibrated_total])[0]
    
    # 6. Output
    print(f"{round(final_calibrated, 2):.2f}")

if __name__ == "__main__":
    main()