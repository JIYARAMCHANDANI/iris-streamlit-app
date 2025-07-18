# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as iris_model.pkl")
