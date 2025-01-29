import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
from datetime import date, datetime
import io

def convert_df_to_csv(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=True)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

# Define the assets with both US and Indian stocks
US_ASSETS = [
    "SPY", "QQQ", "TQQQ", "UPRO", "SOXL","FAS", "IGV","SMH",
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META",
    "NFLX", "NVDA", "TSLA", "AVUV", "ANET", "ORCL",
    "PANW", "COST", "BRK.B", "UNH", "JNJ", "JPM",
    "V", "PG", "MA", "XOM", "HD", "CVX", "ABBV",
    "PFE", "DIS", "CMCSA", "PEP", "CSCO", "MRK",
    "INDA"
]

# Add .NS suffix for NSE stocks and .BO for BSE stocks
INDIAN_ASSETS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "KPITTECH.NS",
    "NIFTYBEES.NS", "UTINEXT50.NS", "RAINBOW.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS",
    "HEALTHIETF.NS", "^NSEBANK", "0P0000ON3O.BO"
]

def get_exchange_calendar(market):
    """Get the appropriate market calendar"""
    if market == "US":
        return mcal.get_calendar('NYSE')
    elif market == "INDIA":
        return mcal.get_calendar('NSE')
    return None

def fetch_data(ticker, start_date, end_date):
    """Fetch data for a single ticker"""
    # Determine market based on ticker suffix
    market = "INDIA" if (".NS" in ticker or ".BO" in ticker) else "US"
    calendar = get_exchange_calendar(market)
    
    if calendar:
        valid_days = calendar.valid_days(start_date=start_date, end_date=end_date)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data = data[data.index.isin(valid_days.date)]
        
        # Convert to INR for Indian stocks
        if market == "INDIA":
            data['Adj Close'] = data['Adj Close'].round(2)
            
        return data
    return pd.DataFrame()

def fetch_market_data(start_date, end_date):
    """Fetch market indices data for both US and Indian markets"""
    market_data = {}
    # US market indices
    us_indices = ['SPY', 'QQQ']
    # Indian market index
    indian_indices = ['^NSEI']
    
    # Fetch US indices
    for ticker in us_indices:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Calculate SMAs
        data['SMA_50'] = ta.sma(data['Adj Close'], length=50)
        data['SMA_150'] = ta.sma(data['Adj Close'], length=150)
        data['SMA_200'] = ta.sma(data['Adj Close'], length=200)
        market_data[ticker] = data
    
    # Fetch Indian index
    for ticker in indian_indices:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Calculate SMAs
        data['SMA_50'] = ta.sma(data['Adj Close'], length=50)
        data['SMA_150'] = ta.sma(data['Adj Close'], length=150)
        data['SMA_200'] = ta.sma(data['Adj Close'], length=200)
        market_data[ticker] = data
    
    return market_data

def check_market_conditions(market_data, date, market="US"):
    """Check if market conditions are favorable based on market indices"""
    if market == "US":
        spy_data = market_data['SPY'].loc[:date]
        qqq_data = market_data['QQQ'].loc[:date]
        
        if len(spy_data) == 0 or len(qqq_data) == 0:
            return False
            
        spy_last = spy_data.iloc[-1]
        qqq_last = qqq_data.iloc[-1]
        
        conditions = [
            spy_last['SMA_50'] > spy_last['SMA_150'],  # SPY 50 SMA > 150 SMA
            qqq_last['SMA_50'] > qqq_last['SMA_150'],  # QQQ 50 SMA > 150 SMA
            spy_last['Adj Close'] > spy_last['SMA_200'],  # SPY price > 200 SMA
            qqq_last['Adj Close'] > qqq_last['SMA_200']   # QQQ price > 200 SMA
        ]
    
    else:  # Indian market
        nsei_data = market_data['^NSEI'].loc[:date]
        
        if len(nsei_data) == 0:
            return False
            
        nsei_last = nsei_data.iloc[-1]
        
        conditions = [
            nsei_last['SMA_50'] > nsei_last['SMA_150'],  # NIFTY 50 SMA > 150 SMA
            nsei_last['Adj Close'] > nsei_last['SMA_200']  # NIFTY price > 200 SMA
        ]
    
    return all(conditions)

def calculate_signals(df, market_data, rsi_entry, rsi_exit, market="US"):
    """Calculate RSI and generate entry/exit signals with market conditions."""
    # Calculate RSI with a 10-day length
    df['RSI_10'] = ta.rsi(df['Adj Close'], length=10)
    
    # Calculate SMAs for both US and Indian stocks
    df['SMA_50'] = ta.sma(df['Adj Close'], length=50)
    df['SMA_150'] = ta.sma(df['Adj Close'], length=150)
    
    # Initialize signals and positions
    df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
    df['Position'] = 0  # 0: no position, 1: in position
    
    # Generate positions
    position = 0
    positions = []
    
    for i in range(len(df)):
        current_date = df.index[i]
        
        # Check stock-specific SMA condition
        sma_50 = df['SMA_50'].iloc[i]
        sma_150 = df['SMA_150'].iloc[i]
        stock_sma_condition = not pd.isna(sma_50) and not pd.isna(sma_150) and sma_50 > sma_150
        
        # Check market conditions based on market type
        market_condition = check_market_conditions(market_data, current_date, market)
        
        # Combined conditions for both markets
        if (df['RSI_10'].iloc[i] <= rsi_entry and 
            stock_sma_condition and 
            market_condition and 
            position == 0):
            position = 1  # Enter position
        elif position == 1 and (
            df['RSI_10'].iloc[i] >= rsi_exit or 
            not stock_sma_condition or 
            not market_condition):
            position = 0  # Exit position
                
        positions.append(position)
    
    df['Position'] = positions
    return df

def calculate_returns(df):
    """Calculate returns and statistics for the strategy."""
    # Ensure the DataFrame is modified explicitly by working on a copy
    df = df.copy()

    # Calculate daily returns
    df.loc[:, 'Daily_Return'] = df['Adj Close'].pct_change()
    
    # Calculate strategy returns (only when in position)
    df.loc[:, 'Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1)
    
    # Calculate cumulative returns
    df.loc[:, 'Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate drawdown
    df.loc[:, 'Peak'] = df['Cumulative_Return'].expanding().max()
    df.loc[:, 'Drawdown'] = (df['Cumulative_Return'] - df['Peak']) / df['Peak'] * 100
    
    # Mark trade entries and exits
    df.loc[:, 'Trade_Entry'] = df['Position'].diff() == 1
    df.loc[:, 'Trade_Exit'] = df['Position'].diff() == -1
    
    return df

def analyze_trades(df, market):
    """Analyze individual trades and calculate statistics with detailed drawdown"""
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

            # Get trade period data
            trade_period = df.loc[entry_date:date]

            # Calculate trade metrics
            trade_return = (exit_price - entry_price) / entry_price * 100
            holding_period = (date - entry_date).days / 30.44  # Convert days to months

            # Calculate trade-specific drawdown
            trade_prices = trade_period['Adj Close']
            trade_peak = trade_prices.expanding().max()
            trade_drawdown = ((trade_prices - trade_peak) / trade_peak * 100)
            max_drawdown = trade_drawdown.min()

            # Calculate high and low prices during trade
            high_price = trade_prices.max()
            low_price = trade_prices.min()

            # Calculate unrealized return at worst point
            worst_return = (low_price - entry_price) / entry_price * 100
            best_return = (high_price - entry_price) / entry_price * 100

            trades.append({
                'Entry_Date': entry_date,
                'Exit_Date': date,
                'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
                'Exit_Price': f"{currency_symbol}{exit_price:.2f}",
                'High_Price': f"{currency_symbol}{high_price:.2f}",
                'Low_Price': f"{currency_symbol}{low_price:.2f}",
                'Return': trade_return,
                'Max_Trade_Drawdown': max_drawdown,
                'Worst_Return': worst_return,
                'Best_Return': best_return,
                'Holding_Period': holding_period,
                'Entry_RSI': df.loc[entry_date, 'RSI_10'],
                'Exit_RSI': row['RSI_10']
            })
            entry_price = None

    # Add last open position if it exists
    if entry_price is not None:
        last_date = df.index[-1]
        last_price = df['Adj Close'].iloc[-1]
        holding_period = (last_date - entry_date).days / 30.44  # Convert days to months

        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': "Open",
            'Entry_Price': f"{currency_symbol}{entry_price:.2f}",
            'Exit_Price': f"{currency_symbol}{last_price:.2f}",
            'High_Price': f"{currency_symbol}{df['Adj Close'][entry_date:].max():.2f}",
            'Low_Price': f"{currency_symbol}{df['Adj Close'][entry_date:].min():.2f}",
            'Return': (last_price - entry_price) / entry_price * 100,
            'Max_Trade_Drawdown': ((df['Adj Close'][entry_date:].min() - entry_price) / entry_price * 100),
            'Worst_Return': ((df['Adj Close'][entry_date:].min() - entry_price) / entry_price * 100),
            'Best_Return': ((df['Adj Close'][entry_date:].max() - entry_price) / entry_price * 100),
            'Holding_Period': holding_period,
            'Entry_RSI': df.loc[entry_date, 'RSI_10'],
            'Exit_RSI': None  # No exit RSI for open positions
        })

    return pd.DataFrame(trades)

def main():
    st.set_page_config(layout="wide")
    st.title("RSI Entry/Exit Strategy Backtester (US & Indian Markets)")

    # Input parameters
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        market = st.selectbox("Select Market", ["US", "India"])
        assets = US_ASSETS if market == "US" else INDIAN_ASSETS
    
    with col2:
        selection_mode = st.radio("Stock Selection", ["From List", "Custom"])
        if selection_mode == "From List":
            ticker = st.selectbox("Select Asset", assets)
        else:
            ticker = st.text_input("Enter Stock Symbol", 
                help="For US stocks, enter symbol (e.g., AAPL). For Indian stocks, add .NS for NSE (e.g., TCS.NS) or .BO for BSE (e.g., TCS.BO)")
    
    with col3:
        end_date = st.date_input("End Date", date.today())
    
    with col4:
        lookback_months = st.slider("Lookback Period (Months)", 1, 120, 60)
    
    with col5:
        rsi_entry = st.number_input("RSI Entry Threshold", value=32, step=1)
        rsi_exit = st.number_input("RSI Exit Threshold", value=79, step=1)

    # Validate custom ticker input
    if selection_mode == "Custom" and ticker:
        if market == "INDIA" and not (ticker.endswith('.NS') or ticker.endswith('.BO')):
            st.warning("For Indian stocks, please add .NS (NSE) or .BO (BSE) suffix to the symbol")
            return
        
    if st.button("Run Analysis"):
        if not ticker:
            st.error("Please enter a stock symbol")
            return
            
        start_date = pd.to_datetime(end_date) - pd.DateOffset(months=lookback_months)
        
        with st.spinner('Analyzing data...'):
            # Fetch stock data
            df = fetch_data(ticker, start_date, end_date)
            
            if df.empty:
                st.error(f"No data available for {ticker}")
                return
            
            # Fetch market data for both US and Indian stocks
            market_data = fetch_market_data(start_date, end_date)
            
            df = calculate_signals(df, market_data, rsi_entry, rsi_exit, market)
            df = calculate_returns(df)
            trades_df = analyze_trades(df, market)
            
            # Display market conditions info based on market
            if market == "US":
                st.info("""
                For US stocks, the following conditions must be met for entry and maintaining positions:
                - Stock's 50-day SMA above 150-day SMA
                - SPY's 50-day SMA above 150-day SMA
                - QQQ's 50-day SMA above 150-day SMA
                - SPY price above its 200-day SMA
                - QQQ price above its 200-day SMA
                - RSI below entry threshold
                
                Positions will exit if any of these conditions are no longer met or if RSI reaches the exit threshold.
                """)
            else:
                st.info("""
                For Indian stocks, the following conditions must be met for entry and maintaining positions:
                - Stock's 50-day SMA above 150-day SMA
                - NIFTY 50's 50-day SMA above 150-day SMA
                - NIFTY 50 price above its 200-day SMA
                - RSI below entry threshold
                
                Positions will exit if any of these conditions are no longer met or if RSI reaches the exit threshold.
                """)
            
            # Display results
            st.subheader("Strategy Performance")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
            num_trades = len(trades_df)
            win_rate = (trades_df['Return'].astype(float) > 0).mean() * 100 if not trades_df.empty else 0
            avg_return = trades_df['Return'].astype(float).mean() if not trades_df.empty else 0
            max_drawdown = df['Drawdown'].min()
            avg_holding = trades_df['Holding_Period'].mean() if not trades_df.empty else 0
            
            col1.metric("Total Return", f"{total_return:.2f}%")
            col2.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            col3.metric("Win Rate", f"{win_rate:.2f}%")
            col4.metric("Avg Holding (Months)", f"{avg_holding:.1f}")
            
            # Display trades
            if not trades_df.empty:
                st.subheader("Individual Trades")
                # Create two columns for trade statistics
                st.subheader("Trade Statistics")
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.markdown("**Return Metrics**")
                    stats_df1 = pd.DataFrame({
                        'Metric': ['Average Return', 'Best Trade Return', 'Worst Trade Return'],
                        'Value': [
                            f"{trades_df['Return'].mean():.2f}%",
                            f"{trades_df['Best_Return'].max():.2f}%",
                            f"{trades_df['Worst_Return'].min():.2f}%"
                        ]
                    })
                    st.dataframe(stats_df1, hide_index=True)
                
                with stat_col2:
                    st.markdown("**Risk Metrics**")
                    stats_df2 = pd.DataFrame({
                        'Metric': ['Average Drawdown', 'Worst Drawdown', 'Avg Holding Period'],
                        'Value': [
                            f"{trades_df['Max_Trade_Drawdown'].mean():.2f}%",
                            f"{trades_df['Max_Trade_Drawdown'].min():.2f}%",
                            f"{trades_df['Holding_Period'].mean():.1f} months"
                        ]
                    })
                    st.dataframe(stats_df2, hide_index=True)
                
                # Display detailed trades table
                st.subheader("Individual Trades")
                st.dataframe(trades_df.style.format({
                    'Return': '{:.2f}%',
                    'Max_Trade_Drawdown': '{:.2f}%',
                    'Worst_Return': '{:.2f}%',
                    'Best_Return': '{:.2f}%',
                    'Holding_Period': '{:.1f}',
                    'Entry_RSI': '{:.2f}',
                    'Exit_RSI': '{:.2f}'
                }))
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    trades_csv = convert_df_to_csv(trades_df)
                    st.download_button(
                        label="Download Trades Data",
                        data=trades_csv,
                        file_name=f'{ticker}_trades.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    full_data_csv = convert_df_to_csv(df)
                    st.download_button(
                        label="Download Full Analysis Data",
                        data=full_data_csv,
                        file_name=f'{ticker}_full_analysis.csv',
                        mime='text/csv'
                    )
            else:
                st.info("No trades were generated during the selected period.")

if __name__ == "__main__":
    main()

