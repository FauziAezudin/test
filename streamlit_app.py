import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Helper function to extract the dataset zip
def extract_zip(zip_path, extract_to_path):
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    st.write("Datasets extracted successfully.")

# Helper function to perform clustering
def perform_clustering(file_path, appliance_name):
    # Load the dataset into a pandas DataFrame
    dataset = pd.read_csv(file_path)
    
    # Extract the features for clustering
    X = np.array(list(zip(dataset['usage_duration_minutes'], dataset['energy_consumption_kWh'])))
    
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Add cluster labels to the DataFrame
    dataset['cluster'] = kmeans.labels_
    
    # Display the clusters in a scatter plot
    cluster_palette = {0: 'red', 1: 'green'}
    st.write(f"### Clusters of Usage Duration and Energy Consumption for {appliance_name}")
    sns.scatterplot(data=dataset, x='usage_duration_minutes', y='energy_consumption_kWh', hue='cluster', palette=cluster_palette)
    st.pyplot()
    
    return kmeans

# Streamlit UI components
st.title("Energy Efficiency Prediction using K-Means Clustering")

# Set the path for your zip file
zip_file_path = "dataset.zip"  # Update this path if necessary
extracted_folder = "datasets"  # Folder where the datasets will be extracted

# Extract the datasets (only once)
if not os.path.exists(extracted_folder):
    extract_zip(zip_file_path, extracted_folder)

# List of appliance files after extraction
appliance_files = {
    "Refrigerator": os.path.join(extracted_folder, "Refrigerator.csv"),
    "HVAC": os.path.join(extracted_folder, "HVAC.csv"),
    "Electronic": os.path.join(extracted_folder, "Electronic.csv"),
    "Dishwasher": os.path.join(extracted_folder, "Dishwasher.csv"),
    "Washing Machine": os.path.join(extracted_folder, "Washingmachine.csv"),
    "Lighting": os.path.join(extracted_folder, "Lighting.csv")
}

# Select appliance
appliance = st.selectbox(
    "Select Appliance",
    ["Refrigerator", "HVAC", "Electronic", "Dishwasher", "Washing Machine", "Lighting"]
)

# Perform clustering based on selected appliance
kmeans = perform_clustering(appliance_files[appliance], appliance)

# Input fields for user to enter energy consumption and usage duration
usage_duration = st.number_input("Enter the usage duration (in minutes):", min_value=1, value=30)
energy_consumption = st.number_input("Enter the energy consumption (in kWh):", min_value=0.1, value=1.0)

# Make prediction based on user input
user_input = [usage_duration, energy_consumption]
cluster = kmeans.predict([user_input])[0]

# Output the corresponding message based on the cluster label
if cluster == 0:
    st.write(f"The {appliance} is **not energy efficient**.")
else:
    st.write(f"The {appliance} is **energy efficient**.")
