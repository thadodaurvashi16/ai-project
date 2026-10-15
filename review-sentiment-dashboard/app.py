import streamlit as st
import pickle


# Page settings
st.set_page_config(
    page_title="Review Sentiment Dashboard",
    page_icon="📝",
    layout="centered"
)


# Load model
with open("models/sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)


# Title
st.title("📝 Review Sentiment Dashboard")

st.write(
    "Enter a customer review and check whether it is Positive or Negative."
)


# Input
review = st.text_area(
    "Enter your review:",
    placeholder="Example: This product is amazing!"
)


# Button
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        # Prediction
        prediction = model.predict([review])[0]

        # Probability
        probability = model.predict_proba([review])[0]

        confidence = probability[prediction] * 100


        # Result
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")


        # Confidence
        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )