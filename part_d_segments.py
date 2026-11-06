"""Part D: hub profile, K-Means segments, separation evidence, tier check."""
import os
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")
PROFILE = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]

profile = (df.groupby("hub_id").agg(shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
                                    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                                    mean_n_stops=("n_stops", "mean"))
           .reset_index().sort_values("hub_id").reset_index(drop=True))  # pinned ascending hub_id order
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

Z = StandardScaler().fit_transform(profile[PROFILE])


def across_over_within(labels):
    across, within = [], []
    for i, j in combinations(range(len(Z)), 2):
        d = float(np.linalg.norm(Z[i] - Z[j]))
        (within if labels[i] == labels[j] else across).append(d)
    return np.mean(across) / np.mean(within)


sep = []
for k in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": k, "across_over_within": round(across_over_within(lab), 4),
                "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep).to_csv("outputs/part_d_separation.csv", index=False)

profile["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Z)
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

sp = (profile.groupby("segment").agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                     delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                     mean_n_stops=("mean_n_stops", "mean"))
      .reindex(["Standard", "Watch", "Escalate"]).reset_index())
sp.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = seg["hub_tier"].map({1: "Standard", 2: "Watch", 3: "Escalate"})
seg[seg["segment"] != implied].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(sp.round(4).to_string(index=False))
