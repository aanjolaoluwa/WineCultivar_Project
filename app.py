from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["ash"]),
            float(request.form["magnesium"]),
            float(request.form["flavanoids"]),
            float(request.form["proline"])
        ]

        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

