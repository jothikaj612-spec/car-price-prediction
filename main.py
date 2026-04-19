import pandas as pd
import os
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("car data.csv")

# =========================
# PREPROCESSING
# =========================
df = df.drop(['Car_Name'], axis=1)

df['Car_Age'] = 2026 - df['Year']
df = df.drop(['Year'], axis=1)

df = pd.get_dummies(df, drop_first=True)

# =========================
# SPLIT DATA
# =========================
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestRegressor()
model.fit(X_train, y_train)

# =========================
# SAVE MODEL (FIXED)
# =========================
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))

# =========================
# STREAMLIT UI
# =========================
st.title("🚗 Car Price Prediction App")

st.write("Enter car details:")

present_price = st.number_input("Present Price", min_value=0.0)
kms_driven = st.number_input("KMs Driven", min_value=0)
owner = st.selectbox("Owner", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
car_age = st.number_input("Car Age", min_value=0)

# =========================
# CONVERT INPUT
# =========================
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
seller_type_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# =========================
# PREDICTION
# =========================
input_dict = {
    'Present_Price': present_price,
    'Kms_Driven': kms_driven,
    'Owner': owner,
    'Car_Age': car_age,
    'Fuel_Type_Diesel': fuel_type_diesel,
    'Seller_Type_Individual': seller_type_individual,
    'Transmission_Manual': transmission_manual
}

input_df = pd.DataFrame([input_dict])

# Add missing columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure same order
input_df = input_df[X.columns]

prediction = model.predict(input_df)