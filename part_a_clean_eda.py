"""Part A: clean the raw export, build features, write checks / delay evidence / correlations."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

dedup = raw.drop_duplicates()
weight_before = pd.to_numeric(dedup["parcel_weight_kg"], errors="coerce")

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

q1, q3 = table["delivery_hours"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(weight_before.isna().sum())),
    ("sentinel_weight_rows", int((weight_before > 100).sum())),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(dedup["service_type"].nunique())),
    ("service_type_spellings_after", int(clean["service_type"].nunique())),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1), ("hours_q3", q3),
    ("hours_lower_bound", lo), ("hours_upper_bound", hi),
    ("hours_outlier_count", int(((table["delivery_hours"] < lo) | (table["delivery_hours"] > hi)).sum())),
    ("declared_value_mean_inr", table["declared_value_inr"].mean()),
    ("declared_value_median_inr", table["declared_value_inr"].median()),
    ("log_declared_value_mean", table["log_declared_value"].mean()),
    ("log_declared_value_median", table["log_declared_value"].median()),
]
pd.DataFrame([(m, round(float(v), 4)) if isinstance(v, (float, np.floating)) else (m, int(v)) for m, v in checks],
             columns=["metric", "value"], dtype=object).to_csv("outputs/part_a_checks.csv", index=False)

table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = table.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for grp, r in g.iterrows():
        rows.append({"dimension": dim, "group": grp, "shipments": int(r["shipments"]), "delayed": int(r["delayed"]),
                     "delay_rate": round(r["delay_rate"], 4), "mean_hours": round(r["mean_hours"], 4)})
pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)

corr = table.drop(columns=["distance_band"]).select_dtypes(include="number").corr().round(4)
corr.index.name = "column"
corr.reset_index().to_csv("outputs/part_a_correlations.csv", index=False)
print(f"modelling table: {len(table)} rows; delayed {int(table['is_delayed'].sum())}")
