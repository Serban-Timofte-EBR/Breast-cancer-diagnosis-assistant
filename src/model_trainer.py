import time
import pickle
import numpy as np
import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

def train_models(X_train, y_train):
    """Train multiple models, compare performance, and select the best one."""
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
    }

    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        
        results.append({
            "Model": name,
            "Mean Accuracy": np.mean(scores),
            "Std Dev": np.std(scores),
            "Training Time": training_time
        })

    results_df = pd.DataFrame(results).sort_values(by="Mean Accuracy", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    print("\n🔹 Model Performance Comparison:\n")
    print(results_df.to_string(index=False))

    print(f"\n✅ Best Model Selected: {best_model_name} with Accuracy: {results_df.iloc[0]['Mean Accuracy']:.4f}")

    # **Handle SHAP Explainer based on Model Type**
    if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        explainer = shap.Explainer(best_model, X_train)
    else:
        print("⚠️ Using SHAP KernelExplainer for Non-Tree-Based Models (e.g., SVM)...")
        explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:50])  # Approximate with 50 samples
    
    shap_values = explainer(X_train[:100])  # Reduce computation time

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    with open("models/shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    print("🚀 Model and SHAP Feature Importance saved!")

    return best_model