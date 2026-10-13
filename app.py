import json
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import train_model as tm

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
if not os.path.exists("models/crop_models.joblib"):
    tm.main()  # train automatically on first run

M = joblib.load("models/crop_models.joblib")
METRICS = json.load(open("models/metrics.json"))
app = Flask(__name__)


def recommend(rain, temp, soil):
    x = pd.DataFrame([{"rainfall_mm": rain, "avg_temp_c": temp, "soil": soil}])
    probs = M["tree"].predict_proba(x)[0]
    classes = M["tree"].classes_
    zone = M["zones"][int(M["km"].predict(x)[0])]
    results = []
    # score every crop: tree suitability + regression yield
    for crop in tm.CROPS:
        yld = max(0.1, float(M["reg"][crop].predict(x)[0]))
        rel = yld / tm.CROPS[crop][5]
        p = float(probs[list(classes).index(crop)]) if crop in classes else 0.0
        results.append({"crop": crop, "suitability": round(p * 100, 1), "yield": round(yld, 2),
                        "unit": "t/ha", "rel": round(rel, 2)})
    # rank: tree suitability first, relative yield breaks ties
    results.sort(key=lambda r: (r["suitability"], r["rel"]), reverse=True)
    return results[:4], zone


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    form = {"rainfall": "", "temp": "", "soil": ""}
    if request.method == "POST":
        form = {k: request.form.get(k, "") for k in form}
        try:
            rain, temp = float(form["rainfall"]), float(form["temp"])
            if not (50 <= rain <= 4000 and 0 <= temp <= 45) or form["soil"] not in tm.SOILS:
                raise ValueError
            result = recommend(rain, temp, form["soil"])
        except ValueError:
            error = "Enter rainfall between 50 and 4000 mm, a temperature between 0 and 45 °C, and choose a soil type."
    return render_template("index.html", soils=tm.SOILS, form=form, result=result, error=error, m=METRICS)


@app.route("/api/recommend")
def api():
    try:
        top, zone = recommend(float(request.args["rainfall"]), float(request.args["temp"]), request.args["soil"])
        return jsonify({"recommendations": top, "zone": zone})
    except (KeyError, ValueError):
        return jsonify({"error": "Provide rainfall, temp and soil"}), 400


if __name__ == "__main__":
    app.run(debug=True)
