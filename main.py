import yfinance as yf
import pandas as pd
import quantstats as qs
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_data(symbol, start_date, end_date):
    """
    Fetch data from Yahoo Finance
    """
    data = yf.download(symbol, 
                      start=start_date,
                      end=end_date,
                      progress=False)
    return data

def calculate_returns(data):
    """
    Calculate daily returns from price data
    """
    returns = data['Adj Close'].pct_change()
    returns = returns.dropna()
    return returns

def generate_full_report(strategy_returns, benchmark_returns, report_name='quantstats_report.html'):
    """
    Generate comprehensive QuantStats report
    """
    # Enable extending pandas functionality
    qs.extend_pandas()
    
    # Generate HTML report
    qs.reports.html(returns=strategy_returns,
                   benchmark=benchmark_returns,
                   output=report_name,
                   title='Trading Strategy Analysis')

def main():
    # Set parameters
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    strategy_symbol = 'AAPL'  # Replace with your strategy returns
    benchmark_symbol = 'SPY'  # Benchmark (S&P 500 ETF)
    
    print(f"Fetching data for {strategy_symbol} and {benchmark_symbol}...")
    
    # Fetch data
    strategy_data = fetch_data(strategy_symbol, start_date, end_date)
    benchmark_data = fetch_data(benchmark_symbol, start_date, end_date)
    
    # Calculate returns
    strategy_returns = calculate_returns(strategy_data)
    benchmark_returns = calculate_returns(benchmark_data)
    
    print("Generating QuantStats report...")
    
    # Generate report
    report_name = f'quantstats_report_{strategy_symbol}_{datetime.now().strftime("%Y%m%d")}.html'
    generate_full_report(strategy_returns, benchmark_returns, report_name)
    
    print(f"Report generated successfully: {report_name}")
    
    # Print some basic statistics
    print("\nBasic Performance Metrics:")
    print(f"Total Return: {(strategy_returns + 1).prod() - 1:.2%}")
    print(f"Annual Volatility: {strategy_returns.std() * (252 ** 0.5):.2%}")
    print(f"Sharpe Ratio: {qs.stats.sharpe(strategy_returns):.2f}")
    print(f"Max Drawdown: {qs.stats.max_drawdown(strategy_returns):.2%}")
    print(f"Calmar Ratio: {qs.stats.calmar(strategy_returns):.2f}")
    print(f"Sortino Ratio: {qs.stats.sortino(strategy_returns):.2f}")

if __name__ == "__main__":
    main()