from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_model(data):
    X = data.drop(columns=['Diagnosis'])
    y = data['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report

if __name__ == "__main__":
    data = pd.read_csv('data/preprocessed_data.csv')
    model, accuracy, report = train_model(data)
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")