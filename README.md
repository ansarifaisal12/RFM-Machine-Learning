# RFM-Machine-Learning
# Project Name: Microsoft Stock Price Prediction Web App
# Overview:
This project aims to build a machine learning model to predict the closing price of Microsoft Corporation stock using historical stock data. The trained model is deployed as a web app using Streamlit, allowing users to input certain features and obtain a prediction of the stock's closing price for a given set of inputs.

# Project Structure:
# Data Collection:

The project uses historical stock data from the 'MSFT.csv' file, containing daily stock information like Open, High, Low, Close, and Volume.
Data Preprocessing:

# Feature Engineering: Lagged features (e.g., 'Close_Lag1') were created to capture historical patterns.
Handling Missing Values: Rows with missing values due to lag features were removed.
# Model Development:

The Random Forest Regressor model was chosen due to its ability to handle non-linearity and perform well in predicting stock prices.
Features like Open, High, Low, Volume, and Close Lag1 were used to train the model.
# Web App Development:

Streamlit, a Python library for creating web applications, was used to deploy the trained model.
The app allows users to input features like Open, High, Low, Volume, and Close Lag1 through sliders and predicts the closing price based on those inputs.
# Evaluation and Deployment:

# Model evaluation metrics such as Mean Squared Error (MSE) and R-squared were used to assess the model's performance.
The trained model was deployed as a web app accessible through a browser using the Streamlit library.
# Project Files:
MSFT.csv: Contains historical stock data for Microsoft Corporation.
stock_price_predictor.py: Python script implementing the machine learning model and Streamlit web app.
README.md: Documentation explaining the project, how to run the app, and other relevant information.
Other dependencies and libraries required for running the project.
<img width="939" alt="Capture" src="https://github.com/ansarifaisal12/RFM-Machine-Learning/assets/115267921/5fcb05f5-817a-4f45-9128-f58e0b89d879">
