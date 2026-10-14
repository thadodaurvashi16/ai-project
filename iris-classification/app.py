# import streamlit as st
# import pickle
# import numpy as np


# # -----------------------------
# # Page Configuration
# # -----------------------------

# st.set_page_config(
#     page_title="Iris Flower Prediction",
#     page_icon="🌸",
#     layout="centered"
# )


# # -----------------------------
# # Load Trained Model
# # -----------------------------

# with open("iris_model.pkl", "rb") as file:
#     model = pickle.load(file)


# # -----------------------------
# # Iris Class Names
# # -----------------------------

# target_names = [
#     "Iris Setosa",
#     "Iris Versicolor",
#     "Iris Virginica"
# ]


# # -----------------------------
# # Title
# # -----------------------------

# st.title("🌸 Iris Flower Prediction")

# st.write(
#     "Enter the measurements of the Iris flower "
#     "to predict its species."
# )


# # -----------------------------
# # Input Fields
# # -----------------------------

# st.subheader("Enter Flower Measurements")


# sepal_length = st.number_input(
#     "Sepal Length (cm)",
#     min_value=0.0,
#     max_value=10.0,
#     value=5.1
# )


# sepal_width = st.number_input(
#     "Sepal Width (cm)",
#     min_value=0.0,
#     max_value=10.0,
#     value=3.5
# )


# petal_length = st.number_input(
#     "Petal Length (cm)",
#     min_value=0.0,
#     max_value=10.0,
#     value=1.4
# )


# petal_width = st.number_input(
#     "Petal Width (cm)",
#     min_value=0.0,
#     max_value=10.0,
#     value=0.2
# )


# # -----------------------------
# # Prediction Button
# # -----------------------------

# if st.button("🔮 Predict Flower"):

#     # Create input data
#     input_data = np.array([
#         [
#             sepal_length,
#             sepal_width,
#             petal_length,
#             petal_width
#         ]
#     ])

#     # Make prediction
#     prediction = model.predict(input_data)

#     # Get flower name
#     predicted_class = target_names[prediction[0]]

#     # Display result
#     st.success(
#         f"Predicted Flower: {predicted_class} 🌸"
#     )





import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# Load Trained Model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Iris Class Names
target_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]

# Title
st.title("🌸 Iris Flower Prediction")

st.write(
    "Enter the measurements of the Iris flower "
    "to predict its species."
)

# Input Fields
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Prediction Button
if st.button("🔮 Predict Flower"):

    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(input_data)

    predicted_class = target_names[int(prediction[0])]

    st.success(
        f"Predicted Flower: {predicted_class} 🌸"
    )
