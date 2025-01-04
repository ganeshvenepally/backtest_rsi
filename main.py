import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import streamlit as st

# Function to calculate RSI using pandas-ta
def calculate_rsi(data, window=10):
    return ta.rsi(data['Close'], length=window)

# Backtest function
def backtest_rsi_strategy(tickers, start_date, end_date, rsi_threshold=32):
    results = []
    for ticker in tickers:
        st.write(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.write(f"No data for {ticker}. Skipping...")
            continue

        data['RSI'] = calculate_rsi(data)
        data['Signal'] = (data['RSI'] < rsi_threshold)

        # Generate buy and sell signals
        data['Buy'] = data['Signal'] & ~data['Signal'].shift(1, fill_value=False)
        data['Sell'] = data['Buy'].shift(1, fill_value=False)

        # Calculate performance
        data['Daily_Return'] = data['Close'].pct_change()
        data['Strategy_Return'] = 0
        data.loc[data['Buy'], 'Strategy_Return'] = data['Daily_Return'].shift(-1)

        total_return = data['Strategy_Return'].sum()
        results.append({
            'Ticker': ticker,
            'Total_Return': total_return,
            'Trades': data['Buy'].sum()
        })

        # Display individual ticker results
        st.write(f"{ticker} Total Return: {total_return:.2%}, Trades: {data['Buy'].sum()}")

    return pd.DataFrame(results)

# Streamlit app
def main():
    st.title("RSI Backtesting Strategy")

    # User inputs
    tickers = st.text_input("Enter tickers (comma-separated):", "AAPL, MSFT, TSLA").split(",")
    start_date = st.date_input("Select start date:", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Select end date:", value=pd.to_datetime("2023-12-31"))
    rsi_threshold = st.slider("RSI Threshold:", min_value=10, max_value=50, value=32)

    if st.button("Run Backtest"):
        results = backtest_rsi_strategy(tickers, start_date, end_date, rsi_threshold)

        # Display results
        st.subheader("Backtest Results")
        st.dataframe(results)

        # Plot results
        st.subheader("Total Return by Ticker")
        if not results.empty:
            fig, ax = plt.subplots()
            results.set_index('Ticker')['Total_Return'].plot(kind='bar', ax=ax, title='Total Return by Ticker')
            ax.set_xlabel('Ticker')
            ax.set_ylabel('Total Return')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
