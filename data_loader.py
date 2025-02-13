import yfinance as yf
import pandas as pd
from datetime import date
import streamlit as st

# Set data range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data  # Caches data to reduce API calls
def load_stock_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            raise ValueError(f"No data found for ticker symbol: {ticker}")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None
