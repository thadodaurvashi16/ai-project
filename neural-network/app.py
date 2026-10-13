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

st.write(
    "Upload a handwritten digit image and the trained "
    "MNIST model will predict the digit."
)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_mnist_model():

    model = load_model("mnist_digit_model.keras")

    return model


model = load_mnist_model()


# ==========================================
# 4. IMAGE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your handwritten digit",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# 5. IF IMAGE IS UPLOADED
# ==========================================

if uploaded_file is not None:

    # --------------------------------------
    # Load uploaded image
    # --------------------------------------

    original_img = Image.open(uploaded_file).convert("L")

    # --------------------------------------
    # Display original image
    # --------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        original_img,
        caption="Your uploaded digit",
        width=250
    )


    # ======================================
    # 6. CONVERT IMAGE TO NUMPY
    # ======================================

    img_array = np.array(original_img)


    # ======================================
    # 7. CREATE DIGIT MASK
    # ======================================

    # White background = 0
    # Black/dark digit = high value

    digit_mask = 255 - img_array


    # ======================================
    # 8. FIND DIGIT PIXELS
    # ======================================

    coords = np.argwhere(digit_mask > 30)


    # ======================================
    # 9. CHECK WHETHER DIGIT EXISTS
    # ======================================

    if coords.size == 0:

        st.error(
            "No digit detected. "
            "Please upload a clear handwritten digit."
        )

        st.stop()


    # ======================================
    # 10. FIND BOUNDING BOX
    # ======================================

    y_min, x_min = coords.min(axis=0)

    y_max, x_max = coords.max(axis=0)


    # ======================================
    # 11. CROP ONLY THE DIGIT
    # ======================================

    cropped = digit_mask[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


    # ======================================
    # 12. CONVERT TO PIL IMAGE
    # ======================================

    cropped_img = Image.fromarray(
        cropped.astype("uint8")
    )


    # ======================================
    # 13. RESIZE DIGIT
    # ======================================

    # Keep original aspect ratio
    # Maximum size = 20 x 20

    cropped_img.thumbnail((20, 20))


    # ======================================
    # 14. CREATE 28 x 28 CANVAS
    # ======================================

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )


    # ======================================
    # 15. CENTER THE DIGIT
    # ======================================

    x_position = (
        28 - cropped_img.width
    ) // 2

    y_position = (
        28 - cropped_img.height
    ) // 2


    # Put digit in center

    canvas.paste(
        cropped_img,
        (x_position, y_position)
    )


    # ======================================
    # 16. CONVERT TO NUMPY ARRAY
    # ======================================

    processed_array = np.array(canvas)


    # ======================================
    # 17. NORMALIZE PIXELS
    # ======================================

    processed_array = (
        processed_array.astype("float32")
        / 255.0
    )


    # ======================================
    # 18. ADD BATCH DIMENSION
    # ======================================

    processed_array = np.expand_dims(
        processed_array,
        axis=0
    )


    # ======================================
    # 19. DISPLAY PROCESSED IMAGE
    # ======================================

    st.subheader("Processed Image")

    st.image(
        processed_array[0],
        caption="28 × 28 MNIST-style image",
        width=250
    )


    # ======================================
    # 20. MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        processed_array,
        verbose=0
    )


    # ======================================
    # 21. FIND PREDICTED DIGIT
    # ======================================

    digit = np.argmax(
        prediction,
        axis=1
    )[0]


    # ======================================
    # 22. FIND CONFIDENCE
    # ======================================

    confidence = (
        np.max(prediction) * 100
    )


    # ======================================
    # 23. DISPLAY PREDICTION
    # ======================================

    st.success(
        f"Predicted Digit: {digit}"
    )


    # ======================================
    # 24. DISPLAY CONFIDENCE
    # ======================================

    st.subheader("Prediction Confidence")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )


    # ======================================
    # 25. DISPLAY ALL PROBABILITIES
    # ======================================

    st.subheader(
        "Prediction Probabilities"
    )


    probabilities = prediction[0] * 100


    for i in range(10):

        st.write(
            f"Digit {i}: "
            f"{probabilities[i]:.2f}%"
        )

        st.progress(
            float(
                probabilities[i] / 100
            )
        )


    # ======================================
    # 26. FINAL MESSAGE
    # ======================================

    st.info(
        f"The model predicted digit "
        f"**{digit}** with "
        f"**{confidence:.2f}% confidence**."
    )