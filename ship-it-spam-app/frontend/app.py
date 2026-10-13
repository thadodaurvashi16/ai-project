# import streamlit as st
# import requests


# # st.set_page_config(
# #     page_title="SMS Spam Detector",
# #     page_icon="📩"
# # )


# st.title("📩 SMS Spam Detector")

# st.write(
#     "Enter an SMS message and check whether it is Spam or Not Spam."
# )


# message = st.text_area(
#     "Enter your message:",
#     height=150
# )


# if st.button("Predict"):

#     if message.strip() == "":
#         st.warning("Please enter a message.")

#     else:

#         try:

#             response = requests.post(
#                 "http://127.0.0.1:8000/predict",
#                 json={"message": message},
#                 timeout=10
#             )

#             if response.status_code == 200:

#                 result = response.json()

#                 prediction = result["prediction"]

#                 if prediction == "Spam":
#                     st.error("🚨 This message is SPAM!")

#                 else:
#                     st.success("✅ This message is NOT SPAM.")

#             else:
#                 st.error("API returned an error.")

#         except requests.exceptions.RequestException:
#             st.error(
#                 "Could not connect to FastAPI. "
#                 "Please make sure the API is running."
#             )


import pickle
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩"
)

st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message and check whether it is Spam or Not Spam.")

# frontend/ થી root ડિરેક્ટરી અને model પાથ મેળવો
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Model files missing at: {MODEL_PATH} or {VECTORIZER_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

message = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        msg_vector = vectorizer.transform([message])
        prediction = model.predict(msg_vector)[0]

        if prediction == 1 or prediction == "Spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")