import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
@st.cache_data
def load_data():
    file_path = 'EnergyDataset.csv'  # Adjust the path to your dataset (if using a local dataset)
    return pd.read_csv(file_path)

# Preprocess the dataset
def preprocess_data(dataset):
    X = dataset[['AnnualEnergyUse', 'ApplianceType']]
    Y = dataset['EnergyEfficiency']
    return train_test_split(X, Y, test_size=0.3, random_state=5)

# Initialize and train the model
@st.cache_resource
def train_model(x_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    return model

# Main Streamlit app
def main():
    # Title and description
    st.title("Energy Efficiency Prediction")
    st.write("This app predicts the energy efficiency of appliances based on annual energy use and appliance type.")
    
    # Load dataset
    dataset = load_data()
    
    # Display dataset info
    if st.checkbox("Show dataset info"):
        st.write("Dataset Columns:")
        st.write(dataset.columns)
        st.write("Dataset Summary:")
        st.write(dataset.info())
    
    # Split data and train the model
    X_train, X_test, y_train, y_test = preprocess_data(dataset)
    model = train_model(X_train, y_train)
    
    # Display R-squared
    st.write(f"R-squared score: {model.score(X_test, y_test):.4f}")
    
    # Prediction section
    st.subheader("Predict Energy Efficiency")
    annual_energy_use = st.number_input("Enter Annual Energy Use (in integer)", min_value=0, max_value=10000, value=1000, step=1)
    appliance_type = st.selectbox("Select Appliance Type", [0, 1, 2, 3], format_func=lambda x: ['Air Conditioner', 'Electric Cooking', 'Clothes Dryer', 'Water Heater'][x])
    
    user_input = pd.DataFrame({
        'AnnualEnergyUse': [annual_energy_use],
        'ApplianceType': [appliance_type]
    })
    
    # Predict energy efficiency
    predicted_efficiency = model.predict(user_input)
    
    # Map to rating
    rating_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    predicted_rating = rating_map.get(int(predicted_efficiency[0]), 'Unknown')
    
    st.write(f"Predicted Energy Efficiency Rating: {predicted_rating}")
    
    # Visualizations
    st.subheader("Visualize Data")
    fig, ax = plt.subplots()
    ax.scatter(dataset['AnnualEnergyUse'], dataset['EnergyEfficiency'], label="Energy Efficiency")
    ax.set_xlabel('Annual Energy Use')
    ax.set_ylabel('Energy Efficiency')
    st.pyplot(fig)
    
    # 3D scatter plot
    st.subheader("3D Visualization")
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(dataset['AnnualEnergyUse'], dataset['ApplianceType'], dataset['EnergyEfficiency'], c=dataset['EnergyEfficiency'], cmap='viridis')
    ax.set_xlabel("Annual Energy Use")
    ax.set_ylabel("Appliance Type")
    ax.set_zlabel("Energy Efficiency")
    cbar = plt.colorbar(sc)
    cbar.set_label('Energy Efficiency')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
