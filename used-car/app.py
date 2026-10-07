# import streamlit as st
# import pandas as pd
# import pickle


# # ==========================================
# # LOAD TRAINED MODEL
# # ==========================================

# with open("used_car_model.pkl", "rb") as file:
#     model = pickle.load(file)


# # ==========================================
# # STREAMLIT PAGE
# # ==========================================

# st.set_page_config(
#     page_title="Used Car Price Prediction",
#     page_icon="🚗",
#     layout="centered"
# )


# st.title("🚗 Used Car Price Prediction")
# st.write("Enter the car details below to predict its price.")


# # ==========================================
# # USER INPUTS
# # ==========================================

# brand = st.text_input(
#     "Brand",
#     value="Toyota"
# )

# model_name = st.text_input(
#     "Model",
#     value="Camry"
# )

# model_year = st.number_input(
#     "Model Year",
#     min_value=1980,
#     max_value=2026,
#     value=2021,
#     step=1
# )

# milage = st.number_input(
#     "Mileage",
#     min_value=0.0,
#     value=30000.0,
#     step=1000.0
# )

# fuel_type = st.selectbox(
#     "Fuel Type",
#     [
#         "Gasoline",
#         "Diesel",
#         "Hybrid",
#         "Electric"
#     ]
# )

# transmission = st.selectbox(
#     "Transmission",
#     [
#         "Automatic",
#         "Manual"
#     ]
# )

# ext_col = st.text_input(
#     "Exterior Color",
#     value="White"
# )

# int_col = st.text_input(
#     "Interior Color",
#     value="Black"
# )

# accident = st.selectbox(
#     "Accident History",
#     [
#         "None reported",
#         "At least 1 accident"
#     ]
# )

# clean_title = st.selectbox(
#     "Clean Title",
#     [
#         "Yes",
#         "No"
#     ]
# )


# # ==========================================
# # FEATURE ENGINEERING
# # ==========================================

# car_age = 2026 - model_year

# mileage_per_year = milage / (car_age + 1)

# engine_liters = st.number_input(
#     "Engine Liters",
#     min_value=0.5,
#     max_value=10.0,
#     value=2.5,
#     step=0.1
# )


# # ==========================================
# # CREATE INPUT DATAFRAME
# # ==========================================

# new_car = pd.DataFrame([{

#     "brand": brand,

#     "model": model_name,

#     "model_year": model_year,

#     "milage": milage,

#     "fuel_type": fuel_type,

#     "transmission": transmission,

#     "ext_col": ext_col,

#     "int_col": int_col,

#     "accident": accident,

#     "clean_title": clean_title,

#     "car_age": car_age,

#     "engine_liters": engine_liters,

#     "mileage_per_year": mileage_per_year

# }])


# # ==========================================
# # PREDICTION BUTTON
# # ==========================================

# if st.button("Predict Car Price"):

#     prediction = model.predict(new_car)

#     predicted_price = prediction[0]

#     st.success(
#         f"Predicted Car Price: ₹{predicted_price:,.2f}"
#     )

#     st.write("### Entered Car Details")

#     st.dataframe(new_car)









import streamlit as st
import pandas as pd
import pickle


# ==========================================
# LOAD MODEL
# ==========================================

with open("used_car_model.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚗 Used Car Price Prediction")
st.write("Enter the car details and predict the estimated price.")


# ==========================================
# INPUTS - ROW 1
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.text_input(
        "Brand",
        value="Toyota"
    )

with col2:
    model_name = st.text_input(
        "Model",
        value="Camry"
    )

with col3:
    model_year = st.number_input(
        "Model Year",
        min_value=1980,
        max_value=2026,
        value=2021,
        step=1
    )


# ==========================================
# INPUTS - ROW 2
# ==========================================

col4, col5, col6 = st.columns(3)

with col4:
    milage = st.number_input(
        "Mileage",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

with col5:
    fuel_type = st.selectbox(
        "Fuel Type",
        [
            "Gasoline",
            "Diesel",
            "Hybrid",
            "Electric"
        ]
    )

with col6:
    transmission = st.selectbox(
        "Transmission",
        [
            "Automatic",
            "Manual"
        ]
    )


# ==========================================
# INPUTS - ROW 3
# ==========================================

col7, col8, col9 = st.columns(3)

with col7:
    ext_col = st.text_input(
        "Exterior Color",
        value="White"
    )

with col8:
    int_col = st.text_input(
        "Interior Color",
        value="Black"
    )

with col9:
    accident = st.selectbox(
        "Accident History",
        [
            "None reported",
            "At least 1 accident"
        ]
    )


# ==========================================
# INPUTS - ROW 4
# ==========================================

col10, col11, col12 = st.columns(3)

with col10:
    clean_title = st.selectbox(
        "Clean Title",
        [
            "Yes",
            "No"
        ]
    )

with col11:
    engine_liters = st.number_input(
        "Engine Liters",
        min_value=0.5,
        max_value=10.0,
        value=2.5,
        step=0.1
    )

with col12:
    st.write("Car Age")
    car_age = 2026 - model_year
    st.info(f"{car_age} years")


# ==========================================
# CALCULATE MILEAGE PER YEAR
# ==========================================

mileage_per_year = milage / (car_age + 1)


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.write("")

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    # ======================================
    # CREATE INPUT DATAFRAME
    # ======================================

    new_car = pd.DataFrame([{

        "brand": brand,

        "model": model_name,

        "model_year": model_year,

        "milage": milage,

        "fuel_type": fuel_type,

        "transmission": transmission,

        "ext_col": ext_col,

        "int_col": int_col,

        "accident": accident,

        "clean_title": clean_title,

        "car_age": car_age,

        "engine_liters": engine_liters,

        "mileage_per_year": mileage_per_year

    }])


    # ======================================
    # PREDICTION
    # ======================================

    prediction = model.predict(new_car)

    predicted_price = prediction[0]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.success(
        f"💰 Predicted Car Price: ₹{predicted_price:,.2f}"
    )


    # ======================================
    # SHOW DETAILS
    # ======================================

    st.subheader("🚘 Car Details")

    st.dataframe(
        new_car,
        use_container_width=True
    )