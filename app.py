from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# load trained model
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get data from form
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]
        return render_template("index.html", result=f"Predicted Diabetes Progression Score: {prediction:.2f}")
    except:
        return jsonify({"error": "Invalid input"})
    
if __name__ == "__main__":
    app.run(debug=True)
