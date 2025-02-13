import streamlit as st
import pandas as pd
import torch
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from data_loader import load_stock_data
from sentiment_analysis import get_financial_news, analyze_sentiment
from lstm_model import train_lstm_model, predict_lstm 




st.title("📈 Financial Forecast & News Viewer")


# Input: Stock Ticker
selected_stock = st.text_input("Enter Ticker Symbol:", "").upper()

if selected_stock:
    st.subheader(f"🔍 Stock Data for {selected_stock}")

    # Load Stock Data
    data = load_stock_data(selected_stock)

    if data is not None:
        data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

        st.markdown(
            """
            <style>
                .dataframe-container {
                    width: 100% !important;  /* Forces full width */
                    max-width: 100% !important;
                }
                .dataframe-container table {
                    width: 100% !important;
                    table-layout: auto !important;
                }
                .dataframe-container th, .dataframe-container td {
                    white-space: nowrap !important;  /* Prevents text wrapping */
                    padding: 10px !important;
                    font-size: 16px !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # ✅ Use st.dataframe() to make sure it expands properly
        st.dataframe(data.tail(), use_container_width=True)


        # Plot Stock Prices
        def plot_raw_data():
            # ✅ Get the latest date in the dataset
            latest_date = pd.to_datetime(data['Date']).max()  # ✅ Ensures it's a datetime object

            # ✅ Set the default starting view (e.g., last 1 year)
            start_display_date = latest_date - pd.DateOffset(years=1)

            # ✅ Create a new figure for historical stock data
            fig = go.Figure()

            # ✅ Add only "Close" prices with a dark blue line
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'], 
                name="Close Prices", line=dict(color="blue")
            ))

            # ✅ Adjust default zoom to last year + future
            fig.update_layout(
                title="Stock Price Trend (Last Year)",
                xaxis_rangeslider_visible=True,
                xaxis=dict(range=[start_display_date, latest_date])
            )

            st.plotly_chart(fig)

        plot_raw_data()

        # 📌 Forecasting Logic (Prophet or LSTM)

        # 📌 Select Prediction Model
        model_choice = st.radio("Select Prediction Model:", ["Prophet", "LSTM"])

        latest_date = pd.to_datetime(data["Date"]).max()

        if model_choice == "Prophet":
            st.subheader("📊 Stock Price Forecast (Prophet)")

            # ✅ Train Prophet Model
            df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
            model = Prophet()
            model.fit(df_train)
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            # ✅ Filter forecast to only show future predictions
            future_forecast = forecast[forecast['ds'] >= latest_date]

            # ✅ Create & Plot Prophet Forecast (No Changes)
            fig_forecast = go.Figure()

            fig_forecast.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'], 
                name="Actual Prices", line=dict(color="blue")
            ))

            fig_forecast.add_trace(go.Scatter(
                x=future_forecast['ds'], y=future_forecast['yhat'], 
                name="Predicted Prices (Prophet)", line=dict(color="orange", dash="dash")
            ))

            fig_forecast.update_layout(
                title="Stock Price Forecast (Prophet)",
                xaxis_rangeslider_visible=True,
                xaxis=dict(range=[latest_date - pd.DateOffset(years=1), future['ds'].max()])
            )

            st.plotly_chart(fig_forecast)

        elif model_choice == "LSTM":
            st.subheader("📊 Stock Price Forecast (LSTM)")

            # ✅ Train LSTM Model
            model, scaler = train_lstm_model(data)

            # ✅ Predict with LSTM
            lstm_forecast = predict_lstm(model, data, scaler, future_days=365)
            future_dates = pd.date_range(start=latest_date, periods=365, freq="D")

            # ✅ Create & Plot LSTM Forecast (Following Prophet's Format)
            fig_lstm = go.Figure()

            fig_lstm.add_trace(go.Scatter(
                x=data['Date'], y=data['Close'], 
                name="Actual Prices", line=dict(color="blue")
            ))

            fig_lstm.add_trace(go.Scatter(
                x=future_dates, y=lstm_forecast, 
                name="Predicted Prices (LSTM)", line=dict(color="green", dash="dash")
            ))

            fig_lstm.update_layout(
                title="Stock Price Forecast (LSTM)",
                xaxis_rangeslider_visible=True,
                xaxis=dict(range=[latest_date - pd.DateOffset(years=1), future_dates.max()])
            )

            st.plotly_chart(fig_lstm)

        # 📌 NEW: Fetch & Display Financial News
        # 📌 Fetch & Display Financial News with GPT-3.5 Sentiment Analysis
        st.subheader("📰 Latest Financial News with Sentiment Analysis")

        news_headlines = get_financial_news(selected_stock)

        if not news_headlines:
            st.write("⚠ No recent news found for this ticker. Try another stock symbol.")
        else:
            sentiment_results = analyze_sentiment(news_headlines)

            for title, url, timestamp, sentiment, explanation in sentiment_results:
                st.markdown(f"**📰 [{title}]({url})**  \n📅 *Published on: {timestamp}*")
                st.markdown(f"📊 **Sentiment:** {sentiment.capitalize()}")
                st.markdown(f"💡 **Analysis:** {explanation}")
                st.write("---")  # Adds a separator


    else:
        st.write("❌ Invalid ticker symbol. Please enter a valid stock ticker.")
