# BTCUSD Trading Bot

This project involves creating a trading bot that predicts the closing price of BTCUSD using an LSTM neural network. The bot collects data, trains a model, and executes trades in real-time. Additionally, it sends updates to a Telegram bot with the actual and predicted prices.
## Table of Contents
1. Overview
2. Installation
3. Data Collection
4. Model Training
5. Telegram Bot Updates
6. Real-Time Prediction and Trading
7. Performance	Metrics

## Overview
This bot is designed to predict the closing price of BTCUSD using an LSTM neural network and execute trades on MetaTrader5. It also sends updates to a Telegram channel using Telegram bot.

## Installation
Ensure you have Python installed, then install the required packages

```python
pip install MetaTrader5 pandas numpy matplotlib seaborn sklearn tensorflow autoviz keras

```

## Data Collection
Set up your MetaTrader5 (MT5) terminal and account credentials. Then run the **Data Collection.ipynb** or run the following script to collect historical data and save it as CSV files. Make sure you saved account into environment variables and choose the path according to your terminal:
Here I have chosen to get past 10,000 rows. feel free to adjust as you want.
```python
import MetaTrader5 as mt5
import pandas as pd
import os

# Path to your MetaTrader5 terminal
terminal_path = r"G:\Program Files\terminal64.exe"

# Initialize MT5
mt5.initialize(path=terminal_path)

# Account credentials
account = os.environ.get('MT5_Account')
password = os.environ.get('MT5_Password')
server = os.environ.get('MT5_Server')

# Log in
mt5.login(account, password=password, server=server)

# Function to collect data
def collect_data(symbol, timeframe, num_bars):
    if not mt5.initialize():
        return None
    if not mt5.symbol_select(symbol, True):
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Collect data
symbol = "BTCUSD"
num_bars = 10000
data_5 = collect_data(symbol, mt5.TIMEFRAME_M5, num_bars)
data_15 = collect_data(symbol, mt5.TIMEFRAME_M15, num_bars)
data_30 = collect_data(symbol, mt5.TIMEFRAME_M30, num_bars)

data_5.to_csv('data_5.csv', index=False)
data_15.to_csv('data_15.csv', index=False)
data_30.to_csv('data_30.csv', index=False)
```

## Model Training
Visualize and make EDA for data using autoviz. choose appropiate data which is suitable according to you and then preprocess the data, then train the LSTM model and save it.
for this run the **Model Training.ipynb** file or run the following scripts:
```python
from autoviz.AutoViz_Class import AutoViz_Class
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# Visualize data
AV = AutoViz_Class()
dft = AV.AutoViz('data_15.csv', sep=",", depVar="close", dfte=None, header=0, verbose=1, lowess=False)

# Load data
data = pd.read_csv('data_15.csv')
data['time'] = pd.to_datetime(data['time'])
data['return'] = data['close'].pct_change()
data['log_return'] = np.log(1 + data['return'])
data.dropna(inplace=True)
X = data[['close', 'log_return', 'open']].values

# Scale data
scaler = MinMaxScaler(feature_range=(0,1)).fit(X)
X_scaled = scaler.transform(X)
y_scaled = [x[0] for x in X_scaled]

# Split data
split = int(len(X_scaled) * 0.85)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# Prepare data for LSTM
timestep = 5
X_train_new, y_train_new = [], []
for step in range(timestep, len(X_train)):
    X_train_new.append(X_train[step - timestep: step])
    y_train_new.append(X_train[step])
X_train_new = np.array(X_train_new)
y_train_new = np.array(y_train_new)

X_test_new, y_test_new = [], []
for step in range(timestep, len(X_test)):
    X_test_new.append(X_test[step - timestep: step])
    y_test_new.append(X_test[step])
X_test_new = np.array(X_test_new)
y_test_new = np.array(y_test_new)

# Build model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_new.shape[1], X_train_new.shape[2])),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(60, activation='tanh', return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(80, activation='tanh', return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(120, activation='tanh'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='linear')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_new, y_train_new, epochs=100, validation_data=(X_test_new, y_test_new), batch_size=32, verbose=1, shuffle=True, callbacks=[early_stopping])
model.save('lstm_model.keras')
```

## Telegram Bot Updates

To set up the Telegram bot for sending updates with actual and predicted prices, follow these steps:

1. Create a Telegram bot using BotFather and obtain the bot token.
2. Replace <YOUR_BOT_API> with your bot's API token in the base_url.
3. Replace <YOUR_CHAT_ID> with your chat ID where you want to receive updates.
4. For giving updates into Telegram channel, add your created bot into that channel.
5. Use the following function to send updates to Telegram:

```python
import requests

# Telegram bot settings
base_url = "https://api.telegram.org/bot<YOUR_BOT_API>/sendMessage"
chat_id = "<YOUR_CHAT_ID>"

# Function to send updates to Telegram
def send_telegram_update(actual_price, predicted_price):
    message = f"Actual price: {actual_price:.2f}\nPredicted close price: {predicted_price:.2f}"
    parameters = {'chat_id': chat_id, 'text': message}
    resp = requests.get(base_url, params=parameters)
    return resp
```

Add this function to your script where the predictions are made and use it to send the updates. This way, you will receive real-time updates on Telegram with the actual and predicted prices.

## Real-Time Prediction and Trading
Load the trained model, collect real-time data according to adjustments, make predictions, send updates to Telegram, and execute trades on MT5
run the **real_time_prediction.ipynb** file or run the following scripts: 

```python
import time
import MetaTrader5 as mt5
from tensorflow.keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import requests

# Load the trained model
model = load_model("lstm_model.keras")

# Define your interval in minutes
interval = 15

# Telegram bot settings
base_url = "https://api.telegram.org/bot<YOUR_BOT_API>/sendMessage"
chat_id = "<YOUR_CHAT_ID>"

# Preprocess data function
def preprocess_data(data, scaler):
    # Converting from object to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Calculating log return
    data['log_return'] = np.log(1 + data['close'].pct_change())

    # Dropping missing values
    data.dropna(inplace=True)

    # Selecting relevant features
    X = data[['close', 'log_return', 'open']]

    # Scaling the features
    X_scaled = scaler.transform(X)

    # Preparing the data for LSTM
    timestep = 5
    X_new = []

    for step in range(timestep, len(X_scaled)):
        X_new.append(X_scaled[step - timestep: step, : X_scaled.shape[1]])

    X_new = np.array(X_new)

    if X_new.size == 0:  # Handle empty X_new
        print("X_new is empty after reshaping. Skipping prediction.")
        return None

    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], X_new.shape[2]))

    return X_new

# Collect latest data function
def collect_latest_data(symbol, timeframe, num_bars=1):
    # Initialize MT5
    if not mt5.initialize():
        print("initialize() failed")
        return None

    # Check if the symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol: {symbol}")
        mt5.shutdown()
        return None

    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to get rates for {symbol}. Error code: {mt5.last_error()}")
        mt5.shutdown()
        return None

    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    
    return data

# Send update to Telegram
def send_telegram_update(actual_price, predicted_price):
    message = f"Actual price: {actual_price:.2f}\nPredicted close price: {predicted_price:.2f}"
    parameters = {
        'chat_id': chat_id,
        'text': message
    }
    resp = requests.get(base_url, params=parameters)
    return resp

# Execute trade function
def execute_trade(predicted_price):
    # Check if MT5 is initialized and reinitialize if necessary
    if not mt5.initialize():
        print("Failed to initialize MT5.")
        return None

    # Ensure the symbol is selected
    if not mt5.symbol_select("BTCUSD", True):
        print("Failed to select symbol BTCUSD.")
        mt5.shutdown()
        return None

    symbol_info_tick = mt5.symbol_info_tick("BTCUSD")
    if symbol_info_tick is None:
        print("Failed to get symbol info tick for BTCUSD.")
        mt5.shutdown()
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "BTCUSD",
        "volume": 0.1,
        "type": mt5.ORDER_TYPE_BUY,
        "price": symbol_info_tick.ask,
        "tp": predicted_price,
        "deviation": 20,
        "magic": 234000,
        "comment": "BTCUSD Prediction",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    mt5.shutdown()
    return result

# Prediction loop
def predict_and_trade():
    scaler = MinMaxScaler(feature_range=(0, 1))

    while True:
        now = datetime.now()
        if now.minute % interval == 0 and now.second == 0:
            # Collect the latest data
            new_data = collect_latest_data("BTCUSD", mt5.TIMEFRAME_M15, 11)  # Adjusted the number of bars as needed for preprocessing

            if new_data is not None and not new_data.empty:
                # Calculate log return
                new_data['log_return'] = np.log(1 + new_data['close'].pct_change())
                new_data.dropna(inplace=True)
                
                # Fit the scaler on the new data including log return
                scaler.fit(new_data[['close', 'log_return', 'open']])

                # Preprocess the data
                X_new = preprocess_data(new_data, scaler)

                if X_new is not None and X_new.shape[0] > 0:
                    # Make a prediction
                    prediction = model.predict(X_new)

                    # Inverse scale the prediction
                    prediction_scaled = np.c_[prediction, np.zeros(prediction.shape), np.zeros(prediction.shape)]
                    prediction_price = scaler.inverse_transform(prediction_scaled)[:, 0]

                    # Get the actual price
                    actual_price = new_data['close'].iloc[-1]

                    # Send update to Telegram
                    send_telegram_update(actual_price, prediction_price[-1])

                    # Execute the trade
                    trade_result = execute_trade(prediction_price[-1])
                    print(f"Trade Result: {trade_result}")

            # Sleep to avoid multiple executions within the same interval
            time.sleep(60)
        time.sleep(1)

# Start the prediction loop
predict_and_trade()
```
## Performance	Metrics
The performance Metrics used during compiling models are:
1. Mean Squared Error
2. Mean Absolute Error
3. Accuracy

Also the r2_scrore applied to measure accuracy which comes with 99.15% and most importantly here used the graph to compare between predicted and actual price on the test set. The whole graphs of all metrics are provided into "Model Training.ipynb" file 
