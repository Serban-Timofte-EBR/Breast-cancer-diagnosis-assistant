from src.data_loader import load_data, preprocess_data
from src.model_trainer import train_models
from src.model_evaluator import evaluate_model
from sklearn.model_selection import train_test_split
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Load & preprocess data
df = load_data("data/breast-cancer.csv")
df = preprocess_data(df)

# Split dataset
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train & evaluate model
best_model = train_models(X_train, y_train)
evaluate_model(best_model, X_test, y_test)

print("✅ AI Model Training Completed!")
