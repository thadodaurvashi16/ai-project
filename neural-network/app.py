import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ==========================================
# 2. TITLE
# ==========================================

st.title("🔢 MNIST Digit Recognition")
st.write("Upload an image of a handwritten digit and the model will predict it.")


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_mnist_model():
    return load_model("mnist_digit_model.keras")


model = load_mnist_model()


# ==========================================
# 4. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your digit image",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# 5. PREDICTION
# ==========================================

if uploaded_file is not None:

    # Load image
    img = Image.open(uploaded_file).convert("L")

    # Display original image
    st.subheader("Uploaded Image")

    st.image(
        img,
        caption="Your uploaded digit",
        width=200
    )

    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    # Resize image to MNIST size
    img = img.resize((28, 28))

    # Convert image to NumPy array
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array.astype("float32") / 255.0

    # Invert image
    # MNIST: black background + white digit
    # User image: usually white background + black digit
    img_array = 1 - img_array

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # ======================================
    # SHOW PROCESSED IMAGE
    # ======================================

    st.subheader("Processed Image")

    st.image(
        img_array[0],
        caption="28 × 28 processed image",
        width=200
    )

    # ======================================
    # PREDICT
    # ======================================

    prediction = model.predict(img_array, verbose=0)

    digit = np.argmax(prediction, axis=1)[0]

    confidence = np.max(prediction) * 100

    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.success(f"Predicted Digit: {digit}")

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    # ======================================
    # ALL DIGIT PROBABILITIES
    # ======================================

    st.subheader("Prediction Probabilities")

    probabilities = prediction[0] * 100

    for i in range(10):
        st.write(f"Digit {i}: {probabilities[i]:.2f}%")
        st.progress(float(probabilities[i] / 100))