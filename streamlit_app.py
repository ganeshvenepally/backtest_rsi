import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
from datetime import date
import io

def convert_df_to_csv(df):
    """Convert DataFrame to CSV."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

# Asset lists
US_ASSETS = [
    "SPY", "QQQ", "TQQQ", "UPRO", "SOXL", "SCHD",
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NFLX", "NVDA", "TSLA"
]
INDIAN_ASSETS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "KPITTECH.NS",
    "NIFTYBEES.NS", "UTINEXT50.NS", "RAINBOW.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BAJAJ-AUTO.NS", "HEALTHIETF.NS", "^NSEBANK", "0P0000ON3O.BO"
]

def fetch_data(ticker, start_date, end_date):
    """Fetch data for a single ticker."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data if not data.empty else pd.DataFrame()

def calculate_signals(df, trend_data, market):
    """Calculate RSI and generate entry/exit signals based on trend conditions."""
    df['RSI_10'] = ta.rsi(df['Adj Close'], length=10)
    df['Signal'] = 0
    df['Position'] = 0

    if market == "US":
        trend_condition = (
            trend_data['QQQ_sma_50'] > trend_data['QQQ_sma_150'] and
            trend_data['SPY_current_price'] > trend_data['SPY_sma_200'] and
            trend_data['QQQ_current_price'] > trend_data['QQQ_sma_200']
        )
        stock_condition = (
            df['Adj Close'].rolling(window=50).mean() > df['Adj Close'].rolling(window=150).mean()
        ) | (
            df['Adj Close'] > df['Adj Close'].rolling(window=200).mean()
        )
        df.loc[trend_condition & stock_condition & (df['RSI_10'] <= 32), 'Signal'] = 1
        df.loc[~trend_condition | ~stock_condition, 'Signal'] = -1
    else:
        df.loc[df['RSI_10'] <= 32, 'Signal'] = 1
        df.loc[df['RSI_10'] >= 79, 'Signal'] = -1

    position = 0
    positions = []
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:
            position = 1
        elif position == 1 and df['Signal'].iloc[i] == -1:
            position = 0
        positions.append(position)
    df['Position'] = positions
    return df

def calculate_returns(df):
    """Calculate returns and statistics for the strategy."""
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1)
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Peak'] = df['Cumulative_Return'].expanding().max()
    df['Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak'] * 100
    df['Trade_Entry'] = df['Position'].diff() == 1
    df['Trade_Exit'] = df['Position'].diff() == -1
    return df

def analyze_trades(df, market):
    """Analyze individual trades and calculate detailed trade statistics."""
    trades = []
    entry_price = None
    entry_date = None
    currency_symbol = "₹" if market == "INDIA" else "$"

    for date, row in df.iterrows():
        if row['Trade_Entry']:
            entry_price = row['Adj Close']
            entry_date = date
        elif row['Trade_Exit'] and entry_price is not None:
            exit_price = row['Adj Close']
            trade_period = df.loc[entry_date:date]
            holding_period = (date - entry_date).days / 30.44
            trade_drawdown = ((trade_period['Adj Close'] - trade_period['Adj Close'].expanding().max()) /
                              trade_period['Adj Close'].expanding().max() * 100).min()

            trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': date,
                'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
                'Exit_Price': f"{currency_symbol}{exit_price:.2f}",
                'Return': (exit_price - entry_price) / entry_price * 100,
                'Max_Trade_Drawdown': trade_drawdown,
                'Holding_Period_Months': holding_period
            })
            entry_price = None

    if entry_price is not None:
        last_date = df.index[-1]
        last_price = df['Adj Close'].iloc[-1]
        holding_period = (last_date - entry_date).days / 30.44
        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': "Open",
            'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
            'Exit_Price': f"{currency_symbol}{last_price:.2f}",
            'Return': (last_price - entry_price) / entry_price * 100,
            'Max_Trade_Drawdown': None,
            'Holding_Period_Months': holding_period
        })

    return pd.DataFrame(trades)

def main():
    st.set_page_config(layout="wide")
    st.title("RSI & Trend-Based Strategy Backtester")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        market = st.selectbox("Select Market", ["US", "India"])
        assets = US_ASSETS if market == "US" else INDIAN_ASSETS
    with col2:
        ticker = st.selectbox("Select Asset", assets)
    with col3:
        end_date = st.date_input("End Date", date.today())
    with col4:
        lookback_months = st.slider("Lookback Period (Months)", 1, 60, 12)

    if st.button("Run Analysis"):
        start_date = pd.to_datetime(end_date) - pd.DateOffset(months=lookback_months)
        df = fetch_data(ticker, start_date, end_date)

        if df.empty:
            st.error(f"No data available for {ticker}")
            return

        trend_data = {}
        if market == "US":
            trend_data = {
                "QQQ_sma_50": yf.download("QQQ", start=start_date, end=end_date)['Adj Close'].rolling(window=50).mean().iloc[-1],
                "QQQ_sma_150": yf.download("QQQ", start=start_date, end=end_date)['Adj Close'].rolling(window=150).mean().iloc[-1],
                "SPY_current_price": yf.download("SPY", start=start_date, end=end_date)['Adj Close'].iloc[-1],
                "SPY_sma_200": yf.download("SPY", start=start_date, end=end_date)['Adj Close'].rolling(window=200).mean().iloc[-1],
                "QQQ_current_price": yf.download("QQQ", start=start_date, end=end_date)['Adj Close'].iloc[-1],
                "QQQ_sma_200": yf.download("QQQ", start=start_date, end=end_date)['Adj Close'].rolling(window=200).mean().iloc[-1],
            }

        df = calculate_signals(df, trend_data, market)
        df = calculate_returns(df)
        trades_df = analyze_trades(df, market)

        st.subheader("Strategy Performance")
        col1, col2, col3, col4 = st.columns(4)
        total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
        num_trades = len(trades_df)
        win_rate = (trades_df['Return'] > 0).mean() * 100 if not trades_df.empty else 0
        avg_return = trades_df['Return'].mean() if not trades_df.empty else 0
        max_drawdown = df['Drawdown'].min()
        avg_holding = trades_df['Holding_Period_Months'].mean() if not trades_df.empty else 0

        col1.metric("Total Return", f"{total_return:.2f}%")
        col2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        col3.metric("Win Rate", f"{win_rate:.2f}%")
        col4.metric("Avg Holding (Months)", f"{avg_holding:.1f}")

        if not trades_df.empty:
            st.subheader("Individual Trades")
            st.dataframe(trades_df)
            trades_csv = convert_df_to_csv(trades_df)
            st.download_button("Download Trades Data", data=trades_csv, file_name=f'{ticker}_trades.csv', mime="text/csv")
        else:
            st.info("No trades were generated during the selected period.")

if __name__ == "__main__":
    main()
