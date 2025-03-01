from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from utils import save_model, load_model
import pandas as pd
from data_preprocessing import load_data, preprocess_data

app = Flask(__name__)

model_filename = 'model/breast_cancer_model.pkl'

model = load_model(model_filename)

if model is None:
    print("Training the model...")
    data = load_data()
    data = preprocess_data(data)
    model, accuracy, report = train_model(data)
    save_model(model, model_filename) 
else:
    print("Model loaded successfully")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = [
            data["radius_mean"],
            data["texture_mean"],
            data["perimeter_mean"],
            data["area_mean"],
            data["smoothness_mean"],
            data["compactness_mean"],
            data["concavity_mean"],
            data["concave_points_mean"],
            data["symmetry_mean"],
            data["fractal_dimension_mean"],
            data["radius_se"],
            data["texture_se"],
            data["perimeter_se"],
            data["area_se"],
            data["smoothness_se"],
            data["compactness_se"],
            data["concavity_se"],
            data["concave_points_se"],
            data["symmetry_se"],
            data["fractal_dimension_se"],
            data["radius_worst"],
            data["texture_worst"],
            data["perimeter_worst"],
            data["area_worst"],
            data["smoothness_worst"],
            data["compactness_worst"],
            data["concavity_worst"],
            data["concave_points_worst"],
            data["symmetry_worst"],
            data["fractal_dimension_worst"]
        ]
        
        prediction = model.predict([input_data])[0]
        result = "Malignant" if prediction == 1 else "Benign"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)