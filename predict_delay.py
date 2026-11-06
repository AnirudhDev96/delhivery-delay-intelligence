"""Score new parcels and turn them into an action list."""
import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
features = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = dict(zip(imp["column"], imp["median"].astype(float)))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians=medians)
feat = build_features(clean, hubs)
out = feat[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(feat[features])[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")
out["action"] = "No action"
out.loc[out["flagged"] == 1, "action"] = "Proactive customer call"
out.loc[(out["flagged"] == 1) & (out["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(f"{len(out)} parcels scored; flagged {int(out['flagged'].sum())}; "
      f"escalations {int((out['action'] == 'Escalate to hub manager').sum())}")
