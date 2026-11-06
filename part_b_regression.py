"""Part B: delivery-hours regression."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(X_train)
train_s, test_s = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame([
    ("train_rows", len(X_train)), ("val_rows", len(X_val)), ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", round(float(np.abs(train_s.mean(axis=0)).max()), 4)),
    ("test_scaled_max_abs_mean", round(float(np.abs(test_s.mean(axis=0)).max()), 4)),
], columns=["metric", "value"], dtype=object).to_csv("outputs/part_b_checks.csv", index=False)


def metrics(name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": name, "split": split, "MAE": round(mean_absolute_error(y_true, y_pred), 4),
            "MSE": round(mse, 4), "RMSE": round(float(np.sqrt(mse)), 4), "R2": round(r2_score(y_true, y_pred), 4)}


rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
rows = [metrics("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

depth_rows, best_depth, best_val = [], None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr, va = rmse(y_train, t.predict(X_train)), rmse(y_val, t.predict(X_val))
    depth_rows.append({"max_depth": "unlimited" if d is None else d, "train_rmse": round(tr, 4), "val_rmse": round(va, 4)})
    if va < best_val:  # strict: ties keep the smaller depth
        best_depth, best_val = d, va
pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, m in candidates.items():
    m.fit(X_train, y_train)
    p = m.predict(X_val)
    rows.append(metrics(name, "val", y_val, p))
    val_rmse[name] = rmse(y_val, p)

lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": round(rmse(y_val, p), 4), "val_r2": round(r2_score(y_val, p), 4)})
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

ranked = sorted(val_rmse, key=val_rmse.get)
X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranked[:2]:
    s = cross_val_score(candidates[name], X_tv, y_tv, cv=kf)
    cv_rows.append({"model": name, "mean_r2": round(s.mean(), 4), "std_r2": round(s.std(), 4)})
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

final = ranked[0]
rows.append(metrics(final, "test", y_test, candidates[final].predict(X_test)))
pd.DataFrame(rows).to_csv("outputs/part_b_metrics.csv", index=False)
print("best tree depth:", best_depth, "| final model:", final)
