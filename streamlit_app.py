import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Title for the Streamlit App
st.title('Energy Efficiency Dashboard')

# Load Dataset
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

# User input
st.subheader("Enter Energy Use and Appliance Type to Predict Energy Efficiency")

annual_energy_use = st.number_input("Enter Annual Energy Use", min_value=0.0)
appliance_type = st.selectbox("Select Appliance Type", [0, 1, 2, 3])

user_input_df = pd.DataFrame({'AnnualEnergyUse': [annual_energy_use], 'ApplianceType': [appliance_type]})

# Prediction
predicted_efficiency = model.predict(user_input_df)
st.write(f"Predicted Energy Efficiency: {predicted_efficiency[0]}")

# R-squared score
st.write(f"R-squared: {model.score(X_test, y_test):.4f}")
