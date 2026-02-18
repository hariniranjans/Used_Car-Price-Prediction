import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained model 
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define feature lists based on your notebook's dummy columns 
BRANDS = ['BMW', 'Bentley', 'Datsun', 'Ferrari', 'Force', 'Ford', 'Honda', 
          'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 
          'Lexus', 'MG', 'Mahindra', 'Maruti', 'Maserati', 'Mercedes-AMG', 
          'Mercedes-Benz', 'Mini', 'Nissan', 'Porsche', 'Renault', 
          'Rolls-Royce', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
FUEL_TYPES = ['Diesel', 'Electric', 'LPG', 'Petrol']

st.title("🚗 Car Price Prediction Interface")
st.markdown("Enter car specifications to estimate the **Selling Price** based on your ML model.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", BRANDS)
    age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5)
    km = st.number_input("Kilometers Driven", min_value=0, value=30000)
    transmission = st.radio("Transmission", ["Manual", "Automatic"])

with col2:
    fuel = st.selectbox("Fuel Type", FUEL_TYPES)
    mileage = st.slider("Mileage (kmpl)", 5.0, 40.0, 18.0)
    engine = st.number_input("Engine Capacity (CC)", min_value=100, max_value=6000, value=1200)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=80.0)

if st.button("Predict Selling Price"):
    # Prepare the input dictionary with all 41 features 
    input_data = {
        'vehicle_age': age,
        'km_driven': km,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power
    }
    
    # Initialize all one-hot encoded brand columns to False 
    for b in BRANDS:
        input_data[f'brand_{b}'] = False
    
    # Initialize fuel type columns to False 
    for f in FUEL_TYPES:
        input_data[f'fuel_type_{f}'] = False
        
    # Initialize transmission 
    input_data['transmission_type_Manual'] = False

    # Set selected categories to True 
    if f'brand_{brand}' in input_data:
        input_data[f'brand_{brand}'] = True
    if f'fuel_type_{fuel}' in input_data:
        input_data[f'fuel_type_{fuel}'] = True
    if transmission == 'Manual':
        input_data['transmission_type_Manual'] = True

    # Convert to DataFrame and align with model features 
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model.feature_names_in_]
    
    # Perform prediction
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Price: ₹{round(prediction, 2):,}")