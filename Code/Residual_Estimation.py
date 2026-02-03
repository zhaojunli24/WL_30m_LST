# Residual Downscaling (Daily) using XGBoost
#
# Author        : Zhao Junli
# Last Modified : 2026-01-29
#
# Description:
# This script trains XGBoost models at 1 km for each DOY (daily residual),
# evaluates training/testing metrics, and predicts 30 m residual for each DOY.
#
# Key features:
# - One model per DOY
# - Consistent preprocessing with ATC script
# - Output prediction is saved as CSV for each DOY
# - Metrics saved as CSV


import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# PART 0. USER SETTINGS


# Paths (1 km training)
train_root_1km = r"H:\WL_30m_LST_V2\Process\XGBoost_1km\Train"

# Paths (30 m prediction input)
input_x_30m_dir = r"H:\WL_30m_LST_V2\Process\XGBoost_30m\Input"

# Paths (30 m prediction output + metrics)
output_root_30m = r"H:\WL_30m_LST_V2\Process\XGBoost_30m\Results_30m1"

# Sample size folder name: N{n_max_1km}
n_max_1km = 15000

# Feature names (must match the CSV column order)
feature_names = [
    "B", "G", "R", "NIR", "SWIR",
    "NDVI", "NDSI", "NDWI",
    "qeff", "snow",
    "elv", "slope", "aspect",
    "SVF", "SSP", "lat", "SAI", "CLCD"
]

# Case settings
case_name = "ALL"
case_feature_idx = list(range(len(feature_names)))  # use all features

# XGBoost parameters
xgb_params = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=5,
    random_state=0,
    n_jobs=8,
    eval_metric="rmse"
)


# PART 1. COMMON UTILITIES (DO NOT EDIT)


def preprocess_features(df):
    """Unified preprocessing for both training and prediction."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if "snow" in df.columns:
        df["snow"] = df["snow"].fillna(0).astype(int).clip(0, 1)

    if "CLCD" in df.columns:
        df["CLCD"] = df["CLCD"].fillna(0).astype(int)

    return df


def dropna_training(x, y):
    """Drop rows containing NaN in x and keep y aligned."""
    valid_mask = ~x.isna().any(axis=1)
    x_out = x.loc[valid_mask, :].copy()
    y_out = y[valid_mask.values]
    return x_out, y_out


def dropna_prediction_keep_length(x):
    """Drop NaN rows for prediction but keep mask for full-length output."""
    valid_mask = ~x.isna().any(axis=1)
    x_valid = x.loc[valid_mask, :].copy()
    return x_valid, valid_mask


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# PART 2. CHECK INPUT DIRECTORIES


input_x_1km_dir = os.path.join(train_root_1km, f"N{n_max_1km}", "X")
input_y_1km_dir = os.path.join(train_root_1km, f"N{n_max_1km}", "Y")

if not os.path.isdir(input_x_1km_dir):
    raise FileNotFoundError(f"Training X folder not found: {input_x_1km_dir}")

if not os.path.isdir(input_y_1km_dir):
    raise FileNotFoundError(f"Training Y folder not found: {input_y_1km_dir}")

os.makedirs(output_root_30m, exist_ok=True)


# PART 3. LIST TRAINING FILES (1 km)


x_files_1km = sorted([
    f for f in os.listdir(input_x_1km_dir)
    if f.startswith("X_") and f.endswith(".csv")
])

n_days_total = len(x_files_1km)

if n_days_total == 0:
    raise RuntimeError(f"No training X csv found in: {input_x_1km_dir}")


# PART 4. MAIN LOOP (TRAIN 1 km -> PREDICT 30 m)


print(f"[INFO] Processing N_max = {n_max_1km}")
print(f"[INFO] Total training days = {n_days_total}")

metrics_records = []
valid_days = 0

output_dir_current = os.path.join(output_root_30m, f"N{n_max_1km}", case_name)
os.makedirs(output_dir_current, exist_ok=True)

for i_day, x_file in enumerate(x_files_1km, start=1):

    # ---- parse DOY from filename: X_001.csv -> 001
    try:
        doy_str = x_file.split("_")[1].split(".")[0]
        doy = int(doy_str)
    except Exception:
        print(f"[WARN] Skip file (cannot parse DOY): {x_file}")
        continue

    x_path_1km = os.path.join(input_x_1km_dir, x_file)
    y_path_1km = os.path.join(input_y_1km_dir, x_file.replace("X_", "Y_"))

    print(f"[INFO] DOY {doy:03d} | {i_day}/{n_days_total}", end="")

    if not os.path.exists(y_path_1km):
        print(" | Y missing")
        continue

    try:
        # ---- read 1km training data
        x_all_1km = pd.read_csv(x_path_1km)

        # strict column check (order matters for reproducibility)
        if list(x_all_1km.columns) != feature_names:
            print(" | 1km columns mismatch")
            continue

        x_all_1km = preprocess_features(x_all_1km)
        x_full_1km = x_all_1km.iloc[:, case_feature_idx].copy()

        y_df_1km = pd.read_csv(y_path_1km)
        y_full_1km = y_df_1km.iloc[:, 0].values

        if len(x_full_1km) != len(y_full_1km):
            print(" | sample mismatch")
            continue

        # drop NaN in training
        x_full_1km, y_full_1km = dropna_training(x_full_1km, y_full_1km)

        # basic sample check
        if len(y_full_1km) < max(10, int(0.5 * n_max_1km)):
            print(" | insufficient training samples")
            continue

        # ---- train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_full_1km, y_full_1km, test_size=0.2, random_state=0
        )

        # ---- train model
        model = XGBRegressor(**xgb_params)
        model.fit(x_train, y_train)

        # ---- evaluate metrics
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        train_rmse = rmse(y_train, pred_train)
        test_rmse = rmse(y_test, pred_test)

        train_r2 = float(r2_score(y_train, pred_train))
        test_r2 = float(r2_score(y_test, pred_test))

        train_bias = float(np.mean(pred_train - y_train))
        test_bias = float(np.mean(pred_test - y_test))

        metrics_records.append({
            "n_max": n_max_1km,
            "case": case_name,
            "doy": doy,
            "train_num": len(y_train),
            "test_num": len(y_test),
            "train_rmse": train_rmse,
            "train_bias": train_bias,
            "train_r2": train_r2,
            "test_rmse": test_rmse,
            "test_bias": test_bias,
            "test_r2": test_r2
        })

        print(f" | test_r2 {test_r2:.3f} | rmse {test_rmse:.3f}", end="")

        # ---- read 30m predictors and predict
        x_file_30m = f"X_30m_{doy:03d}.csv"
        x_path_30m = os.path.join(input_x_30m_dir, x_file_30m)

        if not os.path.exists(x_path_30m):
            print(" | 30m X missing")
            continue

        x_all_30m = pd.read_csv(x_path_30m)

        if list(x_all_30m.columns) != feature_names:
            print(" | 30m columns mismatch")
            continue

        x_all_30m = preprocess_features(x_all_30m)
        x_input_30m = x_all_30m.iloc[:, case_feature_idx].copy()

        # drop NaN for prediction, keep output length
        x_valid_30m, valid_mask = dropna_prediction_keep_length(x_input_30m)
        pred_valid_30m = model.predict(x_valid_30m)

        pred_30m_full = np.full((len(x_input_30m),), np.nan, dtype=float)
        pred_30m_full[valid_mask.values] = pred_valid_30m

        out_csv_name = f"pred_residual_30m_{doy:03d}.csv"
        out_csv_path = os.path.join(output_dir_current, out_csv_name)

        pd.DataFrame({"residual_pred": pred_30m_full}).to_csv(out_csv_path, index=False)

        print(" | saved")
        valid_days += 1

    except Exception as e:
        print(f" | error: {e}")


# PART 5. SAVE METRICS


if metrics_records:
    df_metrics = pd.DataFrame(metrics_records).sort_values("doy")

    out_metrics_csv = os.path.join(
        output_root_30m,
        f"metrics_residual_n{n_max_1km}_{case_name}.csv"
    )

    try:
        df_metrics.to_csv(out_metrics_csv, index=False)
        print(f"\n[INFO] Metrics saved: {out_metrics_csv}")
    except PermissionError:
        print(f"\n[ERROR] Cannot save metrics (file in use): {out_metrics_csv}")
else:
    print("\n[WARN] No valid metrics generated.")

print("[INFO] ALL FINISHED")
