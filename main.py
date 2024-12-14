import streamlit as st
from datetime import date

import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Financial Forcasts")

# input box for the ticker symbol
selected_stock = st.text_input("Enter Ticker Symbol:", "")

# Convert the entered ticker to uppercase to make it case-insensitive
selected_stock = selected_stock.upper()

# Only display the following sections if a ticker is entered
if selected_stock:
    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        try:
            # Attempt to download the stock data
            data = yf.download(ticker, START, TODAY)
            if data.empty:
                raise ValueError(f"No data found for ticker symbol: {ticker}")
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            # Handle errors (e.g., invalid ticker or network issue)
            st.error(f"Error fetching data: {e}")
            return None  # Return None if data cannot be fetched

    # Load data for the selected stock ticker
    data_load_state = st.text("Load data...")
    data = load_data(selected_stock)
    
    if data is not None:
        data_load_state.text("Loading data...done!")

        st.subheader("Raw data")
        st.write(data.tail())

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
            fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Forecasting
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.write('Forecast Plot')
        fig1 = plot_plotly(m, forecast)
        fig1.update_traces(marker=dict(color='darkorange'))
        st.plotly_chart(fig1)

        st.write('Forecast Components')
        fig2 = m.plot_components(forecast)
        st.write(fig2)
    else:
        # If data is None, the error message has already been displayed
        st.write("Please enter a valid stock ticker to begin.")
else:
    st.write("Please enter a stock ticker to begin.")
