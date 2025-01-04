import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

# Function to calculate RSI using pandas-ta
def calculate_rsi(data, window=10):
    return ta.rsi(data['Close'], length=window)

# Backtest function
def backtest_rsi_strategy(tickers, start_date, end_date, rsi_threshold=32):
    results = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data for {ticker}. Skipping...")
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

        # Print or save the trades for the ticker if needed
        print(f"{ticker} Total Return: {total_return:.2%}, Trades: {data['Buy'].sum()}")

    return pd.DataFrame(results)

# Parameters
tickers = ["AAPL", "MSFT", "TSLA"]  # List of stocks or ETFs
start_date = "2020-01-01"
end_date = "2023-12-31"

# Run the backtest
results = backtest_rsi_strategy(tickers, start_date, end_date)

# Display results
print("\nBacktest Results:")
print(results)

# Visualize results
results.set_index('Ticker')['Total_Return'].plot(kind='bar', title='Total Return by Ticker')
plt.xlabel('Ticker')
plt.ylabel('Total Return')
plt.show()