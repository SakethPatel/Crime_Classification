import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
import joblib
from datetime import datetime

# Load trained model and encoders
model = joblib.load("crime_classifier_lgbm.pkl")  
scaler = joblib.load("scaler.pkl")  
ordinal_encoder = joblib.load("ordinal_encoder.pkl")  
label_encoder = joblib.load("label_encoder.pkl")  
target_encodings = joblib.load("target_encodings.pkl")  

train_df = pd.read_csv("train.csv")

# Filter out NaN values from the relevant columns
premise_mapping = train_df[['Premise_Code', 'Premise_Description']].dropna(subset=['Premise_Description']).drop_duplicates().set_index('Premise_Description')['Premise_Code'].to_dict()
area_mapping = train_df[['Area_ID', 'Area_Name']].dropna(subset=['Area_Name']).drop_duplicates().set_index('Area_Name')['Area_ID'].to_dict()
victim_sex_mapping = train_df[['Victim_Sex']].dropna(subset=['Victim_Sex']).drop_duplicates()['Victim_Sex'].to_list()
victim_descent_mapping = train_df[['Victim_Descent']].dropna(subset=['Victim_Descent']).drop_duplicates()['Victim_Descent'].to_list()

# Weapon Description mapping
weapon_mapping = train_df[['Weapon_Used_Code', 'Weapon_Description']].dropna(subset=['Weapon_Description']).drop_duplicates().set_index('Weapon_Description')['Weapon_Used_Code'].to_dict()

# Status Description mapping
status_mapping = train_df[['Status', 'Status_Description']].dropna(subset=['Status_Description']).drop_duplicates().set_index('Status_Description')['Status'].to_dict()

# Function to preprocess user input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])

    # Calculate 'Reported_bins'
    reported_bins = calculate_reported_bins(user_input['Reported_date'], user_input['Occurred_date'])
    df['Reported_bins'] = reported_bins

    # Apply target encoding for 'Location' and 'Modus_Operandi'
    df['Location_encoded'] = df['Location'].map(target_encodings['Location']).fillna(-1)
    df['Modus_Operandi_encoded'] = df['Modus_Operandi'].map(target_encodings['Modus_Operandi']).fillna(-1)

    # One-Hot Encoding (OHE) for categorical features
    categorical_features = ['Reporting_District_no', 'Premise_Code', 'Area_ID', 'Victim_Sex', 'Victim_Descent', 
                            'Weapon_Used_Code', 'Status', 'Reported_bins', 'Time_Occurred_Label']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Calculate 'Reported-Occured' 
    df['Reported-Occured'] = (pd.to_datetime(user_input['Reported_date']) - pd.to_datetime(user_input['Occurred_date'])).days

    # Standard Scaling for numerical features
    df[['Reported-Occured', 'Victim_Age']] = scaler.transform(df[['Reported-Occured', 'Victim_Age']])

    # Apply Ordinal Encoding for 'Part_1-2'
    df['Part_1-2'] = ordinal_encoder.transform(df[['Part_1-2']])

    return df

# Function to calculate reported_bins
def calculate_reported_bins(reported_date, occurred_date):
    try:
        # Parse dates
        reported_date = datetime.strptime(reported_date, "%Y-%m-%d")
        occurred_date = datetime.strptime(occurred_date, "%Y-%m-%d")

        # Ensure occurred_date is not greater than reported_date
        if occurred_date > reported_date:
            raise ValueError("Occurred date cannot be later than reported date.")
        
        # Calculate difference between reported and occurred dates
        diff = (reported_date - occurred_date).days
        
        # Define bins and labels
        bins = [0, 15, 180, 365, float('inf')]
        labels = ['Within 15 days', '15 days to 6 months', '6 months to 1 year', 'Greater than 1 year']
        
        # Assign bin based on the time difference
        reported_bin = pd.cut([diff], bins=bins, labels=labels, right=False, include_lowest=True)
        return reported_bin[0]
    except Exception as e:
        raise ValueError(f"Invalid Date Format: {str(e)}")

# Streamlit UI
st.title("Crime Classification App 🚔")
st.write("Enter crime details to classify the type of crime and view model metrics.")

# User input fields
location = st.selectbox("Location", options=train_df['Location'].dropna().unique())
modus_operandi = st.text_input("Modus Operandi")
reporting_district = st.selectbox("Reporting District No", options=train_df['Reporting_District_no'].dropna().unique())

# Display premise description in the dropdown and map it to premise code
premise_description = st.selectbox("Premise Description", options=list(premise_mapping.keys()))
premise_code = premise_mapping[premise_description]

# Display area name in the dropdown and map it to area ID
area_name = st.selectbox("Area Name", options=list(area_mapping.keys()))
area_id = area_mapping[area_name]

# Display victim sex in the dropdown
victim_sex = st.selectbox("Victim Sex", options=victim_sex_mapping)

# Display victim descent in the dropdown
victim_descent = st.selectbox("Victim Descent", options=victim_descent_mapping)

# Display weapon description in the dropdown and map it to weapon code
weapon_description = st.selectbox("Weapon Description", options=list(weapon_mapping.keys()))
weapon_code = weapon_mapping[weapon_description]

# Display status description in the dropdown and map it to status code
status_description = st.selectbox("Status Description", options=list(status_mapping.keys()))
status_code = status_mapping[status_description]

# Input fields for reporting and occurred dates
reported_date = st.date_input("Reported Date")
occurred_date = st.date_input("Occurred Date")

# Display "Time Occurred Label" dropdown with the updated options
time_occurred_label = st.selectbox("Time Occurred Label", ["Morning", "Afternoon", "Evening", "Night"])

victim_age = st.number_input("Victim Age", min_value=0, max_value=100, value=30)
part_1_2 = st.selectbox("Part 1-2 Classification", [1, 2])

# Predict Button
if st.button("Predict Crime Category"):
    user_input = {
        "Location": location,
        "Modus_Operandi": modus_operandi,
        "Reporting_District_no": reporting_district,
        "Premise_Code": premise_code,  
        "Area_ID": area_id,  
        "Victim_Sex": victim_sex,  
        "Victim_Descent": victim_descent,  
        "Weapon_Used_Code": weapon_code, 
        "Status": status_code,  
        "Reported_date": str(reported_date),
        "Occurred_date": str(occurred_date),
        "Time_Occurred_Label": time_occurred_label,  
        "Victim_Age": victim_age,
        "Part_1-2": part_1_2
    }

    try:
        # Preprocess input
        processed_input = preprocess_input(user_input)

        # Ensure input matches training data columns
        missing_cols = set(model.feature_name()) - set(processed_input.columns)
        for col in missing_cols:
            processed_input[col] = 0  

        processed_input = processed_input[model.feature_name()]  # Align column order

        # Make prediction
        prediction_proba = model.predict(processed_input)
        prediction_class = np.argmax(prediction_proba, axis=1)

        # Decode prediction
        crime_category = label_encoder.inverse_transform(prediction_class)[0]

        st.success(f"Predicted Crime Category: {crime_category}")
      
    except Exception as e:
        st.error(f"An error occurred: {e}")
