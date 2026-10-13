"""Builds the dataset and trains the three data-mining models.

  1. Decision Tree  -> classifies the best crop for rainfall / temperature / soil
  2. Regression     -> predicts crop yield (t/ha), polynomial ridge regression
  3. K-Means        -> clusters growing conditions into agro-climate zones

Run:  python train_model.py
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SOILS = ["Alluvial", "Black", "Red", "Sandy", "Clay", "Loamy"]

# crop: (rain mean mm, rain sd, temp mean C, temp sd, suited soils, base yield t/ha)
CROPS = {
    "Rice":       (1400, 250, 26, 2.5, ["Alluvial", "Clay"],            4.5),
    "Wheat":      (550,  120, 18, 2.5, ["Alluvial", "Loamy"],           3.8),
    "Maize":      (800,  150, 24, 3.0, ["Loamy", "Alluvial", "Red"],    5.0),
    "Cotton":     (700,  130, 28, 2.5, ["Black"],                       2.0),
    "Sugarcane":  (1500, 250, 29, 2.5, ["Alluvial", "Black", "Loamy"], 70.0),
    "Millet":     (400,  100, 30, 2.5, ["Sandy", "Red"],                1.6),
    "Groundnut":  (550,  110, 27, 2.5, ["Sandy", "Red", "Loamy"],       2.2),
    "Chickpea":   (450,  100, 21, 2.5, ["Black", "Loamy"],              1.5),
    "Potato":     (650,  120, 17, 2.5, ["Loamy", "Sandy"],              22.0),
}


def make_dataset(n=4500, seed=42):
    """Synthetic historical crop-yield records (replace with real data if you have it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        crop = rng.choice(list(CROPS))
        rm, rs, tm, ts, soils, base = CROPS[crop]
        rain = max(100, rng.normal(rm, rs))
        temp = rng.normal(tm, ts)
        soil = rng.choice(soils) if rng.random() < 0.85 else rng.choice(SOILS)
        fit = np.exp(-0.5 * ((rain - rm) / (1.6 * rs)) ** 2) * np.exp(-0.5 * ((temp - tm) / (1.6 * ts)) ** 2)
        fit *= 1.0 if soil in soils else 0.7
        yld = max(0.1, base * (0.25 + 0.85 * fit) * rng.normal(1, 0.03))
        rows.append((round(rain, 1), round(temp, 1), soil, crop, round(yld, 2)))
    return pd.DataFrame(rows, columns=["rainfall_mm", "avg_temp_c", "soil", "crop", "yield_t_ha"])


def main():
    df = make_dataset()
    df.to_csv("data/crop_data.csv", index=False)
    print(f"Dataset: {len(df)} rows -> data/crop_data.csv")

    X = df[["rainfall_mm", "avg_temp_c", "soil"]]
    pre = lambda: ColumnTransformer(
        [("soil", OneHotEncoder(categories=[SOILS], handle_unknown="ignore"), ["soil"])],
        remainder="passthrough")

    # 1) Decision tree: conditions -> crop
    Xtr, Xte, ytr, yte = train_test_split(X, df["crop"], test_size=0.2, random_state=1)
    tree = Pipeline([("pre", pre()), ("tree", DecisionTreeClassifier(max_depth=9, min_samples_leaf=8, random_state=1))])
    tree.fit(Xtr, ytr)
    acc = accuracy_score(yte, tree.predict(Xte))

    # 2) Regression: one polynomial ridge model per crop -> yield (t/ha)
    reg, actual, predicted = {}, [], []
    for crop, g in df.groupby("crop"):
        gtr, gte = train_test_split(g, test_size=0.2, random_state=1)
        reg_pre = ColumnTransformer([
            ("soil", OneHotEncoder(categories=[SOILS], handle_unknown="ignore"), ["soil"]),
            ("num", Pipeline([("sc", StandardScaler()), ("poly", PolynomialFeatures(2, include_bias=False))]),
             ["rainfall_mm", "avg_temp_c"])])
        model = Pipeline([("pre", reg_pre), ("ridge", Ridge(alpha=1.0))])
        model.fit(gtr[["rainfall_mm", "avg_temp_c", "soil"]], gtr["yield_t_ha"])
        reg[crop] = model
        actual += list(gte["yield_t_ha"] / CROPS[crop][5])
        predicted += list(model.predict(gte[["rainfall_mm", "avg_temp_c", "soil"]]) / CROPS[crop][5])
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    # 3) K-Means: agro-climate zones
    km_pre = pre()
    km = Pipeline([("pre", km_pre), ("sc", StandardScaler()), ("km", KMeans(n_clusters=5, n_init=10, random_state=1))])
    km.fit(X)
    df["cluster"] = km.predict(X)
    zones = {}
    for c, g in df.groupby("cluster"):
        rain, temp = g.rainfall_mm.mean(), g.avg_temp_c.mean()
        wet = "Wet" if rain > 1000 else "Moderate-rain" if rain > 600 else "Dry"
        heat = "hot" if temp > 27 else "warm" if temp > 22 else "cool"
        top = g.crop.value_counts().head(3).index.tolist()
        zones[int(c)] = {"name": f"{wet}, {heat} zone", "avg_rainfall": round(rain), "avg_temp": round(temp, 1),
                         "main_soil": g.soil.mode()[0], "top_crops": top, "records": int(len(g))}

    joblib.dump({"tree": tree, "reg": reg, "km": km, "zones": zones}, "models/crop_models.joblib")
    json.dump({"tree_accuracy": round(acc * 100, 1), "reg_r2": round(r2, 3), "reg_mae_rel": round(mae, 3),
               "rows": len(df), "zones": zones,
               "crop_counts": df.crop.value_counts().to_dict()}, open("models/metrics.json", "w"), indent=2)
    print(f"Decision Tree accuracy : {acc:.1%}")
    print(f"Regression R2          : {r2:.3f}")
    print(f"K-Means zones          : {len(zones)}")


if __name__ == "__main__":
    main()
