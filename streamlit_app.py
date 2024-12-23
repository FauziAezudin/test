import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Title for the Streamlit App
st.title('Energy Efficiency Dashboard')

# Define the appliance brands list
appliance_brands = [
    'DELLA', 'Friedrich', 'Frigidaire', 'Frigidaire Gallery', 'Hisense', 'Honeywell', 'Hykolity', 
    'Insignia', 'Keplerx', 'Keystone', 'LG', 'LUBECK', 'Midea', 'GE Profile', 'Gradient', 'GREE', 'HEMA', 
    'BLACK+DECKER', 'Century', 'Comfort Aire', 'Danby', 'Noma', 'Noma iQ', 'OMNI MAX', 'Perfect aire',
    'Richmond', 'ROVSUN', 'TCL', 'Vissani', 'Whirlpool', 'Windmill', 'ZOKOP', 'Electrolux', 'Samsung', 
    'Signature Kitchen Suite', 'Maytag', 'Hotpoint', 'GE', 'Bertazzoni', 'Kenmore', 'Inglis', 'Amana', 
    'Blomberg', 'Beko', 'Crosley', 'Asko', 'Miele', 'Speed Queen', 'Bosch', 'Fisher&Paykel', 'Summit', 
    'Magic Chef', 'ELEMENT', 'AEG', 'BREDA', 'BLACK DECKER', 'Avanti', 'FINLUX', 'KOOLMORE', 
    'Equator Advanced Appliances', 'Smad', 'Direct Supply', 'Criterion', 'Marathon', 'LG SIGNATURE', 
    'TECHOMEY', 'A. O. Smith', '1HVAC', 'ACIQ', 'DIYCOOL', 'PolarWave', 'STEALTH', 'American', 
    'RELIANCE WATER HEATERS', 'State', 'Lochinvar', 'Kepler', 'Bradford White', 'JETGLAS', 'SANCO2', 
    'U.S. Craftmaster', 'AMERICAN STANDARD WATER HEATERS', 'Rheem', 'Ruud', 'stream33', 'Hubbell', 'Noritz', 
    'VAUGHN THERMAL', 'AquaThermAire', 'Rinnai', 'Smart Solar'
]

# Fill the lists for ApplianceType and EnergyType with empty strings or placeholders
appliance_types = ['Air conditioner', 'Electric cooking product', 'Clothes dryers', 'Water heater'] + [''] * (len(appliance_brands) - 4)
energy_types = ['electric', 'gas'] + [''] * (len(appliance_brands) - 2)

# Appliance data dictionary
appliance_data = {
    'BN': appliance_brands,
    'ApplianceType': appliance_types,
    'EnergyType': energy_types
}

# Create the DataFrame
appliance_df = pd.DataFrame(appliance_data)

# Display the appliance data (brands and types)
st.write("Appliance Data:", appliance_df)

# Now load the Energy Efficiency dataset
file_path = 'EnergyDataset.csv'  # Replace with actual file path if needed
dataset = pd.read_csv(file_path)

# Display dataset information
st.write("Features (Columns):", dataset.columns)
st.write("Dataset Info:", dataset.info())

# Show a correlation matrix
corr = dataset.corr()
st.write("Correlation Matrix:", corr)

# Plot some graphs
st.subheader("Scatter Plot of Annual Energy Use vs Energy Efficiency")
fig, ax = plt.subplots()
ax.scatter(dataset['AnnualEnergyUse'], dataset['EnergyEfficiency'], marker='o')
ax.set_xlabel('Annual Energy Use')
ax.set_ylabel('Energy Efficiency')
st.pyplot(fig)

# Train the model (you can use your existing code for training here)
X = dataset[['AnnualEnergyUse', 'ApplianceType']]
y = dataset['EnergyEfficiency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# User input: Select appliance and energy type
st.subheader("Enter Energy Use and Appliance Type to Predict Energy Efficiency")

# Display a dropdown for selecting an appliance from the `ApplianceType`
appliance_choice = st.selectbox("Select Appliance Type", appliance_df['ApplianceType'].dropna())

# Map appliance choice to its type (you can add more logic if needed)
appliance_type = appliance_df[appliance_df['ApplianceType'] == appliance_choice].index[0]

# Display additional input for energy use
annual_energy_use = st.number_input("Enter Annual Energy Use", min_value=0.0)

# Map energy type based on selected appliance
energy_type = appliance_df.loc[appliance_type, 'EnergyType'] if appliance_type in appliance_df.index else "Unknown"

st.write(f"Energy Type: {energy_type}")

# Create the user input DataFrame for prediction
user_input_df = pd.DataFrame({'AnnualEnergyUse': [annual_energy_use], 'ApplianceType': [appliance_type]})

# Prediction
predicted_efficiency = model.predict(user_input_df)
st.write(f"Predicted Energy Efficiency: {predicted_efficiency[0]}")

# Define categories for energy efficiency
def categorize_efficiency(efficiency_value):
    if efficiency_value >= 90:
        return 'A'
    elif 75 <= efficiency_value < 90:
        return 'B'
    elif 50 <= efficiency_value < 75:
        return 'C'
    else:
        return 'D'

# Categorize the predicted energy efficiency
efficiency_category = categorize_efficiency(predicted_efficiency[0])

# Display the categorized result
st.write(f"Energy Efficiency Rating: {efficiency_category}")

# R-squared score
st.write(f"R-squared: {model.score(X_test, y_test):.4f}")
