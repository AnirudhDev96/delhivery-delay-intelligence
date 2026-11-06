"""Shared constants and the cleaning / feature functions used by every Part."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
TARGET_REG = "delivery_hours"
TARGET_CLS = "is_delayed"
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100


def parse_declared_value(series):
    """'₹1,290' -> 1290.0; blanks stay NaN."""
    text = series.astype("string").str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce").astype(float)


def clean_shipments(raw, medians=None):
    """Return (clean_df, medians). Steps follow the brief in order."""
    df = raw.drop_duplicates().reset_index(drop=True).copy()                      # 1
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()   # 2
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")                     # 3
    df["declared_value_inr"] = parse_declared_value(df["declared_value"])         # 4
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])                     # 5
    df["parcel_weight_kg"] = pd.to_numeric(df["parcel_weight_kg"], errors="coerce").astype(float)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan  # 6
    if medians is None:                                                           # 7
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    else:
        medians = {c: float(medians[c]) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])
    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and model features. Never uses delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
