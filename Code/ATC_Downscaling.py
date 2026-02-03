# ATC Downscaling (Alpha / Beta) using XGBoost
#
# Author        : Zhao Junli
# Last Modified : 2026-01-29
#
# Description:
# This script trains XGBoost models to downscale ATC parameters (alpha and beta).
# - Train at 1 km using prepared CSV files
# - Predict alpha/beta at 30 m (one-time prediction for annual parameters)
#
# Notes:
# - Output is saved as CSV for better reproducibility on GitHub
# - Unified preprocessing is applied to both training and prediction


import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# PART 0. USER SETTINGS


# Paths
train_root_1km = r"H:\WL_30m_LST_V2\Process\ATC_Downscaling\Train"
input_x_30m_dir = r"H:\WL_30m_LST_V2\Process\ATC_Downscaling\Predict\Input"
output_root = r"H:\WL_30m_LST_V2\Process\ATC_Downscaling\Predict1"

# Training sample sizes (folder name: N{n_train})
n_list = [15000]

# Targets
targets = ["alpha", "beta"]

# Full feature list in CSV (for checking / reference)
feature_names_full = [
    "B", "G", "R", "NIR", "SWIR",
    "NDVI", "qeff",
    "elv", "slope", "aspect",
    "SVF", "SSP", "lat",
    "RSRI", "NDVI_range", "qeff_log_range"
]

# Feature sets
features_alpha = [
    "B", "G", "R", "NIR", "SWIR",
    "NDVI", "qeff",
    "elv", "slope", "aspect",
    "SVF", "SSP", "lat",
    "RSRI"
]

features_beta = feature_names_full

# XGBoost parameters
xgb_params = dict(
    n_estimators=800,
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


# PART 1. COMMON UTILITIES


def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def check_required_columns(df, required_cols, tag="CSV"):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{tag} missing required columns: {missing}")


def preprocess_features(df):

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if "snow" in df.columns:
        df["snow"] = df["snow"].fillna(0).astype(int).clip(0, 1)

    if "CLCD" in df.columns:
        df["CLCD"] = df["CLCD"].fillna(0).astype(int)

    return df


def dropna_training(x, y):

    valid_mask = ~x.isna().any(axis=1)
    x_out = x.loc[valid_mask, :].copy()
    y_out = y[valid_mask.values]
    return x_out, y_out


def dropna_prediction_keep_length(x):

    valid_mask = ~x.isna().any(axis=1)
    x_valid = x.loc[valid_mask, :].copy()
    return x_valid, valid_mask


def get_feature_list(target):
    target = target.lower()
    if target == "alpha":
        return features_alpha
    elif target == "beta":
        return features_beta
    else:
        raise ValueError(f"Unknown target: {target}")


def load_training_data(train_dir, target):
    """
    Expected training files:
      {target.capitalize()}_X.csv
      {target.capitalize()}_Y.csv
    Example:
      Alpha_X.csv, Alpha_Y.csv
      Beta_X.csv,  Beta_Y.csv
    """
    x_path = os.path.join(train_dir, f"{target.capitalize()}_X.csv")
    y_path = os.path.join(train_dir, f"{target.capitalize()}_Y.csv")

    check_file_exists(x_path)
    check_file_exists(y_path)

    x_all = pd.read_csv(x_path)
    y_all = pd.read_csv(y_path).iloc[:, 0].values

    return x_all, y_all


def load_predictors_30m(target):
    """
    Expected prediction file:
      {target}_X_30m.csv
    Example:
      alpha_X_30m.csv
      beta_X_30m.csv
    """
    x30_path = os.path.join(input_x_30m_dir, f"{target}_X_30m.csv")
    check_file_exists(x30_path)
    x30 = pd.read_csv(x30_path)
    return x30


def save_prediction_csv(y_pred_full, out_csv, target):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame({f"{target}_pred": y_pred_full}).to_csv(out_csv, index=False)


# PART 2. MAIN


def run_atc_downscaling():
    os.makedirs(output_root, exist_ok=True)

    for n_train in n_list:
        train_dir = os.path.join(train_root_1km, f"N{n_train}")
        print(f"\n[INFO] ATC Downscaling | N = {n_train}")

        for target in targets:
            try:
                print(f"[INFO] Target: {target}")

                feature_list = get_feature_list(target)

                # ---- load 1km training
                x_all_1km, y_all_1km = load_training_data(train_dir, target)
                check_required_columns(x_all_1km, feature_list, tag=f"1km {target} X")

                # ---- select + preprocess
                x_full_1km = x_all_1km.loc[:, feature_list].copy()
                x_full_1km = preprocess_features(x_full_1km)

                # ---- drop NaN in training
                x_full_1km, y_all_1km = dropna_training(x_full_1km, y_all_1km)

                # ---- train/test split
                x_train, x_test, y_train, y_test = train_test_split(
                    x_full_1km, y_all_1km, test_size=0.2, random_state=0
                )

                # ---- train model
                model = XGBRegressor(**xgb_params)
                model.fit(x_train, y_train, verbose=False)

                # ---- load 30m predictors
                x_30m = load_predictors_30m(target)
                check_required_columns(x_30m, feature_list, tag=f"30m {target} X")

                x_30m_use = x_30m.loc[:, feature_list].copy()
                x_30m_use = preprocess_features(x_30m_use)

                # ---- drop NaN for prediction, but keep output length
                x_30m_valid, valid_mask = dropna_prediction_keep_length(x_30m_use)
                y_pred_valid = model.predict(x_30m_valid)

                y_pred_30m = np.full((len(x_30m_use),), np.nan, dtype=float)
                y_pred_30m[valid_mask.values] = y_pred_valid

                # ---- save output
                out_dir = os.path.join(output_root, "atc_params_30m", f"N{n_train}")
                out_csv = os.path.join(out_dir, f"pred_{target}_30m.csv")
                save_prediction_csv(y_pred_30m, out_csv, target)

                print(f"[INFO] Saved: {out_csv}")

            except Exception as e:
                print(f"[ERROR] {target} failed: {e}")

    print("\n[INFO] ATC Downscaling Finished")


if __name__ == "__main__":
    run_atc_downscaling()
