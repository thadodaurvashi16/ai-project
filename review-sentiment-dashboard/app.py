# import streamlit as st
# import pickle


# # Page settings
# st.set_page_config(
#     page_title="Review Sentiment Dashboard",
#     page_icon="📝",
#     layout="centered"
# )


# # Load model
# with open("models/sentiment_model.pkl", "rb") as file:
#     model = pickle.load(file)


# # Title
# st.title("📝 Review Sentiment Dashboard")

# st.write(
#     "Enter a customer review and check whether it is Positive or Negative."
# )


# # Input
# review = st.text_area(
#     "Enter your review:",
#     placeholder="Example: This product is amazing!"
# )


# # Button
# if st.button("Analyze Sentiment"):

#     if review.strip() == "":
#         st.warning("Please enter a review.")

#     else:
#         # Prediction
#         prediction = model.predict([review])[0]

#         # Probability
#         probability = model.predict_proba([review])[0]

#         confidence = probability[prediction] * 100


#         # Result
#         if prediction == 1:
#             st.success("😊 Positive Review")
#         else:
#             st.error("😞 Negative Review")


#         # Confidence
#         st.write(
#             f"Confidence: **{confidence:.2f}%**"
#         )


import streamlit as st
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Review Sentiment Dashboard",
    page_icon="📝"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file not found!")
    st.write("Expected model path:", str(MODEL_PATH))
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.title("📝 Review Sentiment Dashboard")

review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if review.strip():
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        confidence = probabilities[prediction] * 100

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a review.")