import io
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
# "backend" is the name of the backend container on the shared Docker network
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")
st.write("Predict the sales revenue of a product in a SuperKart store.")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product features
st.markdown("**Product details**")
product_id_char = st.selectbox("Product Category Code (FD = Food, NC = Non-Consumable, DR = Drinks)", ["FD", "NC", "DR"])
product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, step=0.1, value=12.66)
product_allocated_area = st.number_input("Product Allocated Area (ratio of display area)", min_value=0.0, max_value=1.0, step=0.001, value=0.027, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, step=1.0, value=117.08)

# Collect user input for store features
st.markdown("**Store details**")
store_type = st.selectbox("Store Type", ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"])
store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"])
store_age_years = st.number_input("Store Age (in years)", min_value=0, max_value=100, step=1, value=16)

# Convert user input into a dictionary matching the API's expected format
input_data = {
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Product_Id_char': product_id_char,
    'Store_Age_Years': store_age_years,
    'Product_Type_Category': product_type_category
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data)  # Send data to Flask API
        if response.status_code == 200:
            prediction = response.json()['Predicted Sales']
            st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
        else:
            st.error(f"Prediction failed: {response.text}")
    except requests.exceptions.RequestException:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")
st.write("Upload a CSV file with the same 10 feature columns used above to predict sales for multiple products.")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        try:
            file_bytes = uploaded_file.getvalue()
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, file_bytes, "text/csv")}  # Send file to Flask API
            )
            if response.status_code == 200:
                predictions = response.json()
                # Show the uploaded data with a column of predicted sales
                results = pd.read_csv(io.BytesIO(file_bytes))
                results["Predicted_Sales"] = [predictions[str(i)] for i in range(len(results))]
                st.success("Batch predictions completed!")
                st.dataframe(results)
            else:
                st.error(f"Batch prediction failed: {response.text}")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the prediction API.")
