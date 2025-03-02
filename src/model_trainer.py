import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def train_models(X_train, y_train):
    """Train multiple models and select the best one based on accuracy."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        results[name] = {
            "Model": model,
            "Mean Accuracy": np.mean(scores),
            "Std Dev": np.std(scores),
            "Training Time": training_time
        }

    best_model_name = max(results, key=lambda k: results[k]["Mean Accuracy"])
    best_model = results[best_model_name]["Model"]
    print(f"Best Model: {best_model_name} with Accuracy: {results[best_model_name]['Mean Accuracy']:.4f}")

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    return best_model