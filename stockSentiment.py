from typing import List, Dict, Optional, Union, Any
import os
from datetime import datetime, timedelta
from smolagents import CodeAgent, tool
from litellm import completion
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from smolagents import LiteLLMModel
import time
import random
from termcolor import colored
from pyfiglet import Figlet
from tqdm import tqdm

# ASCII art for terminal display
def print_header(text):
    f = Figlet(font='slant')
    header = f.renderText(text)
    print(colored(header, 'cyan'))

def print_section(text):
    print("\n" + "=" * 80)
    print(colored(f" {text} ", 'yellow', attrs=['bold']))
    print("=" * 80)

def print_result(label, value, color='green'):
    print(f"{colored(label, 'white', attrs=['bold'])}: {colored(value, color)}")

def print_progress_bar(description):
    for i in tqdm(range(100), desc=colored(description, 'cyan'), ncols=100):
        time.sleep(random.uniform(0.01, 0.03))

def print_ascii_chart(data, title, width=60, height=10):
    """Create a simple ASCII chart from data"""
    if not data:
        return
    
    values = list(data.values()) if isinstance(data, dict) else data
    labels = list(data.keys()) if isinstance(data, dict) else [str(i) for i in range(len(data))]
    
    max_val = max(values)
    min_val = min(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    print(colored(f"\n{title}", 'yellow', attrs=['bold']))
    print("┌" + "─" * width + "┐")
    
    for i in range(height, 0, -1):
        line = "│"
        for val in values:
            normalized = (val - min_val) / range_val if range_val else 0.5
            pos = int(normalized * height)
            if pos == i - 1:
                line += colored("o", 'red')
            elif pos > i - 1:
                line += "│"
            else:
                line += " "
        line += "│"
        print(line)
    
    print("└" + "─" * width + "┘")
    
    # Print labels
    label_line = " "
    for label in labels:
        label = str(label)[:8]
        padding = max(1, int(width / len(labels)) - len(label))
        label_line += label + " " * padding
    print(label_line)

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
        print_progress_bar(f"Fetching data for {ticker}")
        
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical market data
        hist = stock.history(period=period)
        
        # Calculate key metrics
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            
            # Daily returns for the last 30 days for visualization
            recent_returns = returns.tail(30).to_dict()
            
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
                "recent_returns": recent_returns,
                "date_range": f"{hist.index[0].date()} to {hist.index[-1].date()}"
            }
            
            # Print some attractive output in the terminal
            print_section(f"STOCK ANALYSIS: {data_summary['company_name']} ({ticker})")
            print_result("Current Price", f"${data_summary['current_price']:.2f}")
            print_result("Annual Return", f"{data_summary['price_change_pct']:.2f}%", 
                       'green' if data_summary['price_change_pct'] > 0 else 'red')
            print_result("Volatility", f"{data_summary['volatility']:.2f}%", 'yellow')
            print_result("Sector", data_summary['sector'])
            
            # Display ASCII chart of recent price movement
            recent_prices = dict(list(hist['Close'].tail(20).items()))
            print_ascii_chart(recent_prices, "RECENT PRICE MOVEMENT")
            
            return data_summary
        else:
            print(colored("⚠️ No data found for this ticker", 'red'))
            return {"error": f"No data found for {ticker}"}
    
    except Exception as e:
        print(colored(f"⚠️ Error: {str(e)}", 'red'))
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
        print_section(f"COMPARING STOCKS: {', '.join(tickers)}")
        print_progress_bar("Analyzing comparative performance")
        
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
            
            # Print comparison results
            print_result("COMPARISON RESULTS", "")
            for ticker, data in sorted_returns:
                color = 'green' if data["return_percentage"] > 0 else 'red'
                print_result(f"{data['name']} ({ticker})", 
                           f"{data['return_percentage']:.2f}% | ${data['start_price']:.2f} → ${data['end_price']:.2f}", 
                           color)
            
            # Display ASCII chart of normalized returns
            normalized_final = {ticker: all_data[ticker].iloc[-1] for ticker in all_data}
            print_ascii_chart(normalized_final, "RELATIVE PERFORMANCE (Starting at 100)")
            
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
            print(colored("⚠️ No data found for the provided tickers", 'red'))
            return {"error": "No data found for the provided tickers"}
    
    except Exception as e:
        print(colored(f"⚠️ Error: {str(e)}", 'red'))
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
        print_progress_bar(f"Calculating technical indicators for {ticker}")
        
        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            print(colored("⚠️ No data found for this ticker", 'red'))
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
        
        # Print technical analysis results
        print_section(f"TECHNICAL ANALYSIS: {stock.info.get('shortName', ticker)} ({ticker})")
        
        signal_color = 'green' if current_position.startswith('Buy') else ('red' if current_position.startswith('Sell') else 'yellow')
        print_result("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
        print_result(f"{short_window}-Day SMA", f"${hist['SMA_Short'].iloc[-1]:.2f}")
        print_result(f"{long_window}-Day SMA", f"${hist['SMA_Long'].iloc[-1]:.2f}")
        print_result("Current Signal", current_position, signal_color)
        print_result("Last Signal", f"{last_signal_type} on {last_signal_date}", 
                    'green' if last_signal_type == 'Buy' else 'red')
        
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
        print(colored(f"⚠️ Error: {str(e)}", 'red'))
        return {"error": str(e)}


@tool
def simulate_market_scenarios(tickers: List[str], scenarios: List[str] = ["recession", "inflation", "techBoom"]) -> Dict:
    """Simulates how a portfolio of stocks would perform under different market scenarios.
    
    Args:
        tickers: List of stock ticker symbols in the portfolio
        scenarios: List of market scenarios to simulate (default: recession, inflation, techBoom)
    
    Returns:
        Dictionary with simulation results and recommendations
    """
    try:
        print_header("STRESS TEST")
        print_section(f"PORTFOLIO STRESS TEST: {', '.join(tickers)}")
        print_progress_bar("Initializing Monte Carlo simulation")
        
        # Get historical data for each ticker
        portfolio_data = {}
        portfolio_weights = {}
        sector_exposure = {}
        
        # For demo purposes, assign random weights (in a real implementation, you'd ask for weights)
        total_weight = len(tickers)
        for ticker in tickers:
            weight = 1  # Equal weight for simplicity
            portfolio_weights[ticker] = weight / total_weight
        
        # Process each stock
        for ticker in tickers:
            print_progress_bar(f"Analyzing historical performance of {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")  # Use 5 years to capture various market conditions
            
            if not hist.empty:
                portfolio_data[ticker] = {
                    "name": stock.info.get('shortName', ticker),
                    "sector": stock.info.get('sector', 'Unknown'),
                    "returns": hist['Close'].pct_change().dropna(),
                    "volatility": hist['Close'].pct_change().std() * np.sqrt(252),
                    "weight": portfolio_weights[ticker]
                }
                
                # Track sector exposure
                sector = stock.info.get('sector', 'Unknown')
                if sector in sector_exposure:
                    sector_exposure[sector] += portfolio_weights[ticker]
                else:
                    sector_exposure[sector] = portfolio_weights[ticker]
        
        # Scenario definitions with sector impacts
        # In a real implementation, these would be based on historical data and economic models
        scenario_impacts = {
            "recession": {
                "Technology": -0.25,
                "Consumer Cyclical": -0.40,
                "Financial Services": -0.35,
                "Communication Services": -0.20,
                "Industrials": -0.30,
                "Real Estate": -0.40,
                "Energy": -0.35,
                "Basic Materials": -0.25,
                "Consumer Defensive": -0.10,
                "Healthcare": -0.15,
                "Utilities": -0.05,
                "Unknown": -0.25
            },
            "inflation": {
                "Technology": -0.15,
                "Consumer Cyclical": -0.25,
                "Financial Services": 0.05,
                "Communication Services": -0.10,
                "Industrials": -0.05,
                "Real Estate": 0.10,
                "Energy": 0.30,
                "Basic Materials": 0.20,
                "Consumer Defensive": -0.10,
                "Healthcare": -0.05,
                "Utilities": 0.10,
                "Unknown": -0.05
            },
            "techBoom": {
                "Technology": 0.45,
                "Consumer Cyclical": 0.20,
                "Financial Services": 0.15,
                "Communication Services": 0.30,
                "Industrials": 0.15,
                "Real Estate": 0.10,
                "Energy": 0.05,
                "Basic Materials": 0.10,
                "Consumer Defensive": 0.05,
                "Healthcare": 0.20,
                "Utilities": 0.00,
                "Unknown": 0.15
            }
        }
        
        # Run simulations for each scenario
        simulation_results = {}
        overall_risk_score = 0
        
        for scenario in scenarios:
            print_progress_bar(f"Simulating {scenario} scenario")
            
            # Calculate expected portfolio return for this scenario based on sector impacts
            expected_return = 0
            for ticker, data in portfolio_data.items():
                sector = data["sector"]
                impact = scenario_impacts[scenario].get(sector, scenario_impacts[scenario]["Unknown"])
                # Add some random variation to make it realistic
                stock_expected_return = impact + (random.uniform(-0.1, 0.1) * data["volatility"])
                expected_return += stock_expected_return * data["weight"]
            
            # Calculate confidence/probability of positive returns
            # This is a simplified model - real models would be more sophisticated
            confidence = max(0, min(100, 50 + (expected_return * 100)))
            
            # Determine survivability rating
            if expected_return > 0.15:
                survivability = "very high"
            elif expected_return > 0:
                survivability = "high"
            elif expected_return > -0.15:
                survivability = "moderate"
            else:
                survivability = "low"
            
            # Add to risk score calculation
            # In a real implementation, this would be more sophisticated
            if scenario == "recession":
                overall_risk_score += (100 - confidence) * 0.5  # Recession has higher weight
            else:
                overall_risk_score += (100 - confidence) * 0.25
                
            simulation_results[scenario] = {
                "expectedReturn": expected_return * 100,  # Convert to percentage
                "confidence": confidence,
                "survivability": survivability
            }
        
        # Normalize risk score to 0-100
        overall_risk_score = min(100, overall_risk_score / len(scenarios))
        
        # Generate recommendations based on simulation results
        recommendations = []
        
        # Add recession hedges if needed
        if simulation_results.get("recession", {}).get("survivability") in ["low", "moderate"]:
            recommendations.append({
                "action": "hedge",
                "instrument": "TLT",  # Treasury ETF
                "allocation": 0.15,
                "rationale": "Add Treasury protection against recession risk"
            })
        
        # Add inflation hedges if needed
        if simulation_results.get("inflation", {}).get("survivability") in ["low", "moderate"]:
            recommendations.append({
                "action": "hedge",
                "instrument": "GLD",  # Gold ETF
                "allocation": 0.10,
                "rationale": "Add Gold exposure to hedge against inflation"
            })
        
        # Identify overexposed sectors
        overexposed_sectors = [sector for sector, exposure in sector_exposure.items() 
                               if exposure > 0.30 and sector != "Unknown"]
        
        if overexposed_sectors:
            recommendations.append({
                "action": "diversify",
                "sectors": overexposed_sectors,
                "rationale": f"Reduce exposure to {', '.join(overexposed_sectors)} sector(s)"
            })
        
        # Print simulation results
        print_section("STRESS TEST RESULTS")
        
        risk_color = 'green'
        if overall_risk_score > 75:
            risk_color = 'red'
        elif overall_risk_score > 50:
            risk_color = 'yellow'
            
        print_result("Portfolio Risk Score", f"{overall_risk_score:.1f}/100", risk_color)
        
        # Print scenario outcomes
        print("\n" + colored("SCENARIO OUTCOMES", 'yellow', attrs=['bold']))
        for scenario, result in simulation_results.items():
            color = 'green' if result["expectedReturn"] > 0 else 'red'
            print_result(f"  {scenario.capitalize()}", 
                       f"{result['expectedReturn']:.1f}% expected return | {result['survivability']} survivability", 
                       color)
        
        # Print recommendations
        if recommendations:
            print("\n" + colored("RECOMMENDATIONS", 'yellow', attrs=['bold']))
            for rec in recommendations:
                if rec["action"] == "hedge":
                    print_result(f"  Add {rec['instrument']}", 
                               f"{rec['allocation']*100:.1f}% allocation - {rec['rationale']}")
                elif rec["action"] == "diversify":
                    print_result(f"  Diversify", rec['rationale'])
        
        # Create an ASCII visualization of scenario outcomes
        scenario_returns = {s: simulation_results[s]["expectedReturn"] for s in simulation_results}
        print_ascii_chart(scenario_returns, "EXPECTED RETURNS BY SCENARIO")
        
        return {
            "portfolioRisk": overall_risk_score,
            "sectorExposure": sector_exposure,
            "scenarioOutcomes": simulation_results,
            "recommendations": recommendations
        }
    
    except Exception as e:
        print(colored(f"⚠️ Error: {str(e)}", 'red'))
        return {"error": str(e)}


# Initialize the agent with the enhanced tools
model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    temperature=0.2,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# Initialize the agent with all tools including new stress testing
agent = CodeAgent(
    tools=[get_stock_data, compare_stocks, calculate_moving_averages, simulate_market_scenarios],
    model=model,
    additional_authorized_imports=[
        "datetime", "yfinance", "pandas", "matplotlib", "numpy", 
        "litellm", "time", "random", "termcolor", "pyfiglet", "tqdm"
    ]
)

def main():
    print_header("AI STOCK ANALYZER")
    print(colored("\nWelcome to the AI-powered Stock Portfolio Analyzer", 'cyan'))
    print(colored("Built with Claude 3.5 Sonnet and Python", 'cyan'))
    print("\nWhat would you like to analyze today?")
    
    query = input(colored("\n> ", 'green'))
    print("\n")
    
    # Run the agent with the user query
    agent.run(query)
    
    print(colored("\nAnalysis complete! Follow me on LinkedIn for more AI finance tools.", 'cyan'))

if __name__ == "__main__":
    main()