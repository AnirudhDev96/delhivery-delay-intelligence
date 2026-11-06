"""Part C: delay classifier, feature signal, tuning and alert threshold."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "accuracy": round(acc, 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# Task 2 - baseline and three classifiers at 0.5
model_rows = [{"model": "blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]
logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **cm_row(y_val, logreg.predict(X_val))})
best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1:
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row(y_val, tree.predict(X_val))})
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **cm_row(y_val, rf.predict(X_val))})
pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)

# Task 3 - feature signal
skb = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": skb.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs[["f_score", "rf_importance"]] = fs[["f_score", "rf_importance"]].round(4)
fs.to_csv("outputs/part_c_feature_scores.csv", index=False)
sk_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    sk_rows.append({"k": k, "val_f1": round(f1_score(y_val, p.predict(X_val)), 4)})
pd.DataFrame(sk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

# Task 4 - tuning on 7 selected columns (selector fitted on training rows only)
K = 7
sel = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xva7 = sel.transform(X_train), sel.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)
default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
fmt = lambda p: str({k: p[k] for k in ["n_estimators", "max_depth", "min_samples_leaf"]})
tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": len(gs.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(gs.cv_results_["params"]) * 5, "best_params": fmt(gs.best_params_),
     "best_cv_accuracy": round(gs.best_score_, 4), "val_f1": round(f1_score(y_val, gs.best_estimator_.predict(Xva7)), 4)},
    {"search": "randomized_search", "candidates": len(rs.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(rs.cv_results_["params"]) * 5, "best_params": fmt(rs.best_params_),
     "best_cv_accuracy": round(rs.best_score_, 4), "val_f1": round(f1_score(y_val, rs.best_estimator_.predict(Xva7)), 4)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": fmt(default_params), "best_cv_accuracy": np.nan,
     "val_f1": round(f1_score(y_val, default7.predict(Xva7)), 4)},
])
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

# Task 5 - choose final configuration (ties: default, grid, randomized)
params_by = {"grid_search": gs.best_params_, "randomized_search": rs.best_params_,
             "default_forest_same_7_features": default_params}
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
vf = dict(zip(tuning["search"], tuning["val_f1"]))
chosen = max(order, key=lambda s: (vf[s], -order.index(s)))
cp = {k: params_by[chosen][k] for k in ["n_estimators", "max_depth", "min_samples_leaf"]}
final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, **cp))]).fit(X_train, y_train)

# Task 6 - threshold
proba_val = final.predict_proba(X_val)[:, 1]
th_rows = [{"threshold": t, **cm_row(y_val, (proba_val >= t).astype(int))}
           for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
th = pd.DataFrame(th_rows)
th.to_csv("outputs/part_c_thresholds.csv", index=False)
ok = th[th["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# Task 7 - one test evaluation
proba_test = final.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_row(y_test, (proba_test >= threshold).astype(int))}]).to_csv(
    "outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": cp["n_estimators"],
               "max_depth": "unlimited" if cp["max_depth"] is None else cp["max_depth"],
               "min_samples_leaf": cp["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen, cp, "threshold:", threshold)
