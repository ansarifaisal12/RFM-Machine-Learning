import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your MSFT.csv dataset
df = pd.read_csv('MSFT.csv')

# Feature engineering: creating lag features
df['Close_Lag1'] = df['Close'].shift(1)  # Lag 1 day

# Removing rows with missing values created by lagging
df.dropna(inplace=True)

# Selecting features and target variable
features = ['Open', 'High', 'Low', 'Volume', 'Close_Lag1']
target = 'Close'

X = df[features]
y = df[target]

# Initializing and fitting the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Streamlit web app
st.title('Microsoft Stock Price Predictor')

st.sidebar.header('User Input Features')

def user_input_features():
    open_price = st.sidebar.slider('Open Price', float(df['Open'].min()), float(df['Open'].max()), float(df['Open'].mean()))
    high_price = st.sidebar.slider('High Price', float(df['High'].min()), float(df['High'].max()), float(df['High'].mean()))
    low_price = st.sidebar.slider('Low Price', float(df['Low'].min()), float(df['Low'].max()), float(df['Low'].mean()))
    volume = st.sidebar.slider('Volume', float(df['Volume'].min()), float(df['Volume'].max()), float(df['Volume'].mean()))
    close_lag1 = st.sidebar.slider('Close Lag1', float(df['Close_Lag1'].min()), float(df['Close_Lag1'].max()), float(df['Close_Lag1'].mean()))
    data = {'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Volume': volume,
            'Close_Lag1': close_lag1}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Making predictions
prediction = rf_model.predict(input_df)

st.subheader('Predicted Close Price')
st.write(prediction)
