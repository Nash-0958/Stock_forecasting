import yfinance as yf
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import io
import base64
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load stock symbols from a CSV file
STOCKS = pd.read_csv('stock-list.csv')
STOCKS = STOCKS[['Symbol', 'Company Name']].dropna()
STOCKS.columns = ['symbol', 'name']  # Renaming columns for consistency
STOCKS = STOCKS.set_index('symbol')['name'].to_dict()

def get_stock_suggestions(query):
    query_upper = query.upper()
    suggestions = [f"{name} ({symbol})" for symbol, name in STOCKS.items() if query_upper in symbol]
    return suggestions

# Layout of the Dash app
app.layout = html.Div([
    html.Div([
        html.H1(id='header-text', children='STOCKyy', className='logo'),
        html.Div([
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': f"{name} ({symbol})", 'value': symbol} for symbol, name in STOCKS.items()],
                placeholder='Start typing stock ticker...',
                multi=False
            ),
            html.Button('Submit', id='submit-button', n_clicks=0)
        ], className='dropdown-container'),
    ], id='initial-container', className='initial-center'),  # Initially centered

    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dcc.Graph(id='stock-forecast-graph', className='hidden'),  # Initially hidden
            html.Div(id='error-message'),
            html.Div(id='stock-table-container', children=[
                dash_table.DataTable(
                    id='stock-table',
                    columns=[
                        {'name': 'Date', 'id': 'Date'},
                        {'name': 'Open', 'id': 'Open'},
                        {'name': 'Close', 'id': 'Close'},
                        {'name': 'Change', 'id': 'Change'},
                        {'name': 'Volume', 'id': 'Volume'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'}
                ),
                html.Button("Download CSV", id="download-button", n_clicks=0),
                dcc.Download(id="download-dataframe-csv")
            ])
        ]
    ),
], className='main-container')

# Callback for stock forecasting and layout adjustments
@app.callback(
    [Output('stock-forecast-graph', 'figure'),
     Output('stock-forecast-graph', 'className'),
     Output('error-message', 'children'),
     Output('initial-container', 'className'),
     Output('loading', 'className'),
     Output('stock-table', 'data'),
     Output('stock-table-container', 'style')],
    [Input('submit-button', 'n_clicks')],
    [dash.State('ticker-dropdown', 'value')]
)
def update_forecast(n_clicks, ticker):
    if n_clicks > 0 and ticker:
        try:
            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Fetch stock data from Yahoo Finance up to the current date
            df = yf.download(f'{ticker}.NS', start="2019-07-30", end=current_date)
            
            if df.empty:
                return ({}, 'hidden', f"No data returned for ticker symbol '{ticker}'. Check the symbol or date range.", 'after-submit', 'hidden', [], {'display': 'none'})

            df['Date'] = df.index
            df1 = df['Close'].values.reshape(-1, 1)

            # Data preprocessing
            scaler = MinMaxScaler(feature_range=(0, 1))
            df1 = scaler.fit_transform(df1)

            # Split the data
            training_size = int(len(df1) * 0.65)
            test_size = len(df1) - training_size
            train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :]

            def create_dataset(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset) - time_step - 1):
                    a = dataset[i:(i + time_step), 0]
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            time_step = 100
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Create the LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(1)
            ])
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=0)

            # Make predictions
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            # Prepare for future prediction
            last_time_step = df1[-time_step:]
            future_predictions = []

            for _ in range(30):  # Forecast for next 30 days
                x_input = last_time_step.reshape(1, time_step, 1)
                prediction = model.predict(x_input)
                future_predictions.append(prediction[0, 0])
                last_time_step = np.append(last_time_step[1:], prediction, axis=0)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

            # Generate future stock data (dummy values for the example)
            future_data = pd.DataFrame({
                'Date': future_dates,
                'Open': np.random.uniform(low=1000, high=2000, size=30),  # Replace with actual data
                'Close': future_predictions.flatten(),
                'Change': np.random.uniform(low=-10, high=10, size=30),  # Replace with actual data
                'Volume': np.random.randint(low=1000, high=1000000, size=30)  # Replace with actual data
            })

            # Plotting
            look_back = 100
            trainPredictPlot = np.empty_like(df1)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

            testPredictPlot = np.empty_like(df1)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

            # Get the stock name for the title
            stock_name = STOCKS.get(ticker, ticker)  # Use ticker if name is not found

            fig = {
                'data': [
                    {'x': df.index, 'y': scaler.inverse_transform(df1).flatten(), 'mode': 'lines', 'name': 'Original'},
                    {'x': df.index[look_back:len(train_predict) + look_back], 'y': train_predict.flatten(), 'mode': 'lines', 'name': 'Train Predict'},
                    {'x': df.index[len(train_predict) + (look_back * 2) + 1:len(df1) - 1], 'y': test_predict.flatten(), 'mode': 'lines', 'name': 'Test Predict'},
                    {'x': future_dates, 'y': future_predictions.flatten(), 'mode': 'lines', 'name': 'Future Predictions', 'line': {'dash': 'dash'}}
                ],
                'layout': {
                    'title': f'{stock_name} Price Forecasting (Next 30 Days)',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Price'},
                }
            }

            return fig, 'visible', "", 'after-submit', 'hidden', future_data.to_dict('records'), {'display': 'block'}

        except Exception as e:
            return ({}, 'hidden', f"An error occurred: {str(e)}", 'after-submit', 'hidden', [], {'display': 'none'})

    return {}, 'hidden', "", 'initial-center', 'hidden', [], {'display': 'none'}

# Callback for exporting table data as CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-button", "n_clicks")],
    [dash.State('stock-table', 'data')]
)
def download_csv(n_clicks, data):
    if n_clicks > 0:
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, "stock_data.csv")
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
