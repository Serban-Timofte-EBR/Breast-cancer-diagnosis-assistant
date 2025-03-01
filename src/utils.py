import joblib
import os

def save_model(model, filename):
    """ Save the trained model to a file. """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """ Load the trained model from a file. """
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"Error: The model file {filename} does not exist.")
        return None