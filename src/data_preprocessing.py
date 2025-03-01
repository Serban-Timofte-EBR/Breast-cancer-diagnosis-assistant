import pandas as pd

def load_data():
    column_names = [
        "ID", "Diagnosis", "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
        "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
        "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE", "Compactness SE", "Concavity SE",
        "Concave Points SE", "Symmetry SE", "Fractal Dimension SE", "Radius Worst", "Texture Worst", "Perimeter Worst",
        "Area Worst", "Smoothness Worst", "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst",
        "Fractal Dimension Worst"
    ]
    
    data = pd.read_csv('../data/wdbc.data', header=None, names=column_names, delim_whitespace=True)
    
    return data

def preprocess_data(data):
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
    data = data.drop(columns=['ID'])
    
    return data

if __name__ == "__main__":
    file_path = 'data/wdbc.data'
    data = load_data(file_path)
    data = preprocess_data(data)
    print(data.head())