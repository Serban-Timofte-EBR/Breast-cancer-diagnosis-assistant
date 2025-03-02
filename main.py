from src.data_loader import load_data, preprocess_data
from src.model_trainer import train_models
from src.model_evaluator import evaluate_model
from sklearn.model_selection import train_test_split
import os

os.makedirs("models", exist_ok=True)

df = load_data("data/breast-cancer.csv")
df = preprocess_data(df)

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = train_models(X_train, y_train)
evaluate_model(best_model, X_test, y_test)

print("✅ AI Model Training Completed!")