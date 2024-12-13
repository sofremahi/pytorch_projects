import pandas as pd
#code to understand the data and labels dimensions for computing tensors
# Load the dataset
file_path = 'bc.data'  
data = pd.read_csv(file_path, header=None)

data = data.dropna()
num_columns = data.shape[1]
columns = [
    "ID", "Diagnosis", "Radius", "Texture", "Perimeter", "Area", "Smoothness",
    "Compactness", "Concavity", "ConcavePoints", "Symmetry", "FractalDimension",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "ConcavePoints_SE", "Symmetry_SE",
    "FractalDimension_SE", "Radius_Worst", "Texture_Worst", "Perimeter_Worst",
    "Area_Worst", "Smoothness_Worst", "Compactness_Worst", "Concavity_Worst",
    "ConcavePoints_Worst", "Symmetry_Worst", "FractalDimension_Worst"
][:num_columns] 

data.columns = columns

# Map Diagnosis to numeric values: M -> 1, B -> 0
data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})
if "ID" in data.columns:
    data = data.drop("ID", axis=1)

# Features (X) and target (y)
X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values 

print(f"Features shape: {X.shape}, Target shape: {y.shape}")
