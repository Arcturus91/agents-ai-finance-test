from typing import List, Dict, Optional, Union
import os
from datetime import datetime, timedelta
from smolagents import CodeAgent, tool
from litellm import completion
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from smolagents import LiteLLMModel

@tool
def get_stock_data(ticker: str, period: str = "1y") -> Dict:
    """Fetches historical stock data for a given ticker.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT, GOOG)
        period: Time period to fetch - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
               Defaults to "1y" (1 year of data)
    
    Returns:
        Dictionary containing stock information and historical data summary
    """
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical market data
        hist = stock.history(period=period)
        
        # Calculate key metrics
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            
            data_summary = {
                "ticker": ticker,
                "company_name": info.get('shortName', 'Unknown'),
                "sector": info.get('sector', 'Unknown'),
                "current_price": hist['Close'].iloc[-1],
                "price_change_pct": ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100,
                "avg_daily_volume": hist['Volume'].mean(),
                "volatility": returns.std() * np.sqrt(252) * 100,  # Annualized volatility in percentage
                "highest_price": hist['High'].max(),
                "lowest_price": hist['Low'].min(),
                "price_data": hist['Close'].to_dict(),
                "date_range": f"{hist.index[0].date()} to {hist.index[-1].date()}"
            }
            
            return data_summary
        else:
            return {"error": f"No data found for {ticker}"}
    
    except Exception as e:
        return {"error": str(e)}


@tool
def compare_stocks(tickers: List[str], period: str = "1y") -> Dict:
    """Compares multiple stocks over a given time period.
    
    Args:
        tickers: List of stock ticker symbols to compare
        period: Time period to fetch - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
               Defaults to "1y" (1 year of data)
    
    Returns:
        Dictionary with comparison results
    """
    try:
        results = {}
        all_data = {}
        start_prices = {}
        
        # Collect data for all tickers
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if not hist.empty:
                # Normalize prices to start at 100 for fair comparison
                start_price = hist['Close'].iloc[0]
                start_prices[ticker] = start_price
                normalized = (hist['Close'] / start_price) * 100
                all_data[ticker] = normalized
                
                # Calculate returns
                total_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                results[ticker] = {
                    "name": stock.info.get('shortName', ticker),
                    "start_price": start_price,
                    "end_price": hist['Close'].iloc[-1],
                    "return_percentage": total_return
                }
        
        # Find best and worst performers
        if results:
            sorted_returns = sorted(results.items(), key=lambda x: x[1]["return_percentage"], reverse=True)
            best_ticker, best_data = sorted_returns[0]
            worst_ticker, worst_data = sorted_returns[-1]
            
            return {
                "stocks": results,
                "best_performer": {
                    "ticker": best_ticker,
                    "name": best_data["name"],
                    "return_percentage": best_data["return_percentage"]
                },
                "worst_performer": {
                    "ticker": worst_ticker,
                    "name": worst_data["name"],
                    "return_percentage": worst_data["return_percentage"]
                }
            }
        else:
            return {"error": "No data found for the provided tickers"}
    
    except Exception as e:
        return {"error": str(e)}


@tool
def calculate_moving_averages(ticker: str, period: str = "1y", short_window: int = 20, long_window: int = 50) -> Dict:
    """Calculates short and long-term moving averages for a stock.
    
    Args:
        ticker: The stock ticker symbol
        period: Time period for data - "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        short_window: Number of days for short-term moving average (default: 20)
        long_window: Number of days for long-term moving average (default: 50)
    
    Returns:
        Dictionary with moving average information and signals
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Calculate moving averages
        hist['SMA_Short'] = hist['Close'].rolling(window=short_window, min_periods=1).mean()
        hist['SMA_Long'] = hist['Close'].rolling(window=long_window, min_periods=1).mean()
        
        # Generate buy/sell signals
        hist['Signal'] = 0
        hist['Signal'][short_window:] = np.where(
            hist['SMA_Short'][short_window:] > hist['SMA_Long'][short_window:], 1, 0
        )
        hist['Position'] = hist['Signal'].diff()
        
        # Find buy/sell points
        buy_signals = hist[hist['Position'] == 1]
        sell_signals = hist[hist['Position'] == -1]
        
        # Current position
        current_position = "Hold"
        if len(hist) > long_window:
            if hist['SMA_Short'].iloc[-1] > hist['SMA_Long'].iloc[-1]:
                current_position = "Buy/Hold" if hist['Position'].iloc[-1] != 1 else "Recently Crossed Above (Buy Signal)"
            else:
                current_position = "Sell/Avoid" if hist['Position'].iloc[-1] != -1 else "Recently Crossed Below (Sell Signal)"
        
        # Last signal
        last_signal_date = "None"
        last_signal_type = "None"
        if not buy_signals.empty and not sell_signals.empty:
            last_buy = buy_signals.index[-1]
            last_sell = sell_signals.index[-1]
            
            if last_buy > last_sell:
                last_signal_date = last_buy.strftime('%Y-%m-%d')
                last_signal_type = "Buy"
            else:
                last_signal_date = last_sell.strftime('%Y-%m-%d')
                last_signal_type = "Sell"
        elif not buy_signals.empty:
            last_signal_date = buy_signals.index[-1].strftime('%Y-%m-%d')
            last_signal_type = "Buy"
        elif not sell_signals.empty:
            last_signal_date = sell_signals.index[-1].strftime('%Y-%m-%d')
            last_signal_type = "Sell"
        
        return {
            "ticker": ticker,
            "company_name": stock.info.get('shortName', ticker),
            "current_price": hist['Close'].iloc[-1],
            f"SMA_{short_window}": hist['SMA_Short'].iloc[-1],
            f"SMA_{long_window}": hist['SMA_Long'].iloc[-1],
            "current_position": current_position,
            "last_signal": {
                "date": last_signal_date,
                "type": last_signal_type
            },
            "short_window": short_window,
            "long_window": long_window
        }
    
    except Exception as e:
        return {"error": str(e)}

model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    temperature=0.2,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)


# Initialize the agent with the finance tools
agent = CodeAgent(
    tools=[get_stock_data, compare_stocks, calculate_moving_averages],
    model=model,  # Using Claude via litellm
    additional_authorized_imports=["datetime", "yfinance", "pandas", "matplotlib", "numpy", "litellm"]
)


agent.run("Should I buy Tesla stocks?")