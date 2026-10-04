# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Keep the JSON keys in the order they were added (row 0, 1, 2, ...)
superkart_api.json.sort_keys = False

# Load the trained machine learning model (preprocessing + Random Forest pipeline)
model = joblib.load("superkart_model.joblib")

# Features expected by the model, in the order used during training
feature_columns = list(model.feature_names_in_)


# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"


# Define an endpoint for single product prediction (POST request)
@superkart_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales as a JSON response.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Check that all required features are present
    missing = [col for col in feature_columns if col not in product_data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_char': product_data['Product_Id_char'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }

    # Convert the extracted data into a Pandas DataFrame (columns in training order)
    input_data = pd.DataFrame([sample])[feature_columns]

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert the NumPy float to a Python float so jsonify can serialize it
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales
    return jsonify({'Predicted Sales': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details for multiple products
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Check that all required features are present
    missing = [col for col in feature_columns if col not in input_data.columns]
    if missing:
        return jsonify({'error': f'Missing columns: {missing}'}), 400

    # Arrange the columns in the order used during training
    input_data = input_data[feature_columns]

    # Make predictions for all products in the DataFrame
    predicted_sales = [round(float(sale), 2) for sale in model.predict(input_data)]

    # Create a dictionary of predictions with the row index as keys
    output_dict = {str(i): sale for i, sale in enumerate(predicted_sales)}

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(debug=True)
