# app.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.datasets import load_iris

# Load the trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Predictor")

st.write("Enter flower measurements to predict the species.")

# User input
sepal_length = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)',
                                       'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(input_data)[0]
    species = load_iris().target_names[prediction]
    st.success(f"The predicted Iris species is: **{species}**")

    st.write("🔍 Prediction input preview:")
    st.dataframe(input_data)
