import streamlit as st
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to churn or stay."
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=2
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# BILLING INFORMATION
# =========================================================

st.divider()

st.subheader("💰 Billing Information")

col4, col5 = st.columns(2)


with col4:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=85.0,
        step=1.0
    )


with col5:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=170.0,
        step=1.0
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    new_customer = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges

    }])


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(new_customer)[0]

    probability = model.predict_proba(
        new_customer
    )[0][1]


    churn_percentage = probability * 100


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    st.divider()

    st.subheader("📈 Prediction Result")


    col6, col7 = st.columns(2)


    with col6:

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )


    with col7:

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to CHURN"
            )

        else:

            st.success(
                "✅ Customer is likely to STAY"
            )


    # -----------------------------------------------------
    # PROGRESS BAR
    # -----------------------------------------------------

    st.write("Churn Risk")

    st.progress(
        min(int(churn_percentage), 100)
    )


    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    if prediction == 1:

        st.warning(
            "This customer has a higher probability of leaving "
            "the company. Consider offering retention benefits "
            "or a suitable plan."
        )

    else:

        st.info(
            "This customer has a lower probability of leaving "
            "the company and is likely to continue the service."
        )


    # -----------------------------------------------------
    # SHOW CUSTOMER DATA
    # -----------------------------------------------------

    with st.expander("View Customer Information"):

        st.dataframe(
            new_customer,
            use_container_width=True
        )