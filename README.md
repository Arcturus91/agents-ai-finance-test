# AI-Powered Stock Analysis Tools

A collection of advanced stock analysis tools powered by AI to help with investment decisions. This toolkit leverages Claude 3.5 Sonnet for intelligent analysis of financial markets.

## Overview

This project provides a suite of AI-enhanced financial analysis tools that can:
- Analyze individual stocks and their performance
- Compare multiple stocks across various metrics
- Calculate technical indicators like moving averages
- Simulate market scenarios and stress test portfolios
- Find high-growth companies based on hedge fund purchases

## Features

### 1. Stock Data Analysis
- Historical price data analysis
- Volatility calculations
- Performance metrics
- Visual representation of stock movements

### 2. Technical Analysis
- Moving average calculations
- Buy/sell signal generation
- Technical indicator-based recommendations

### 3. Portfolio Comparison
- Side-by-side analysis of multiple stocks
- Normalized performance comparison
- Best/worst performer identification

### 4. Market Scenario Simulation
- Stress testing portfolios against recession, inflation, tech boom scenarios
- Portfolio risk score calculation
- Sector exposure analysis
- Hedging recommendations

### 5. Hedge Fund Strategy Analysis
- Identify stocks recently purchased by hedge funds
- Revenue growth analysis of potential investments
- Focus on high-growth companies (30%+ growth)

## Prerequisites

- Python 3.8+
- An Anthropic API key for Claude access
- A Polygon.io API key (for the revenue growth tool)

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install smolagents litellm yfinance pandas numpy matplotlib termcolor pyfiglet tqdm python-dotenv requests
```

3. Set up your environment variables:

```bash
# Create a .env file with your API keys
ANTHROPIC_API_KEY=your_anthropic_api_key
POLYGON_API_KEY=your_polygon_api_key  # Only needed for revenue growth analysis
```

## Usage

### Basic Stock Analysis

```python
from financeTest import agent as finance_agent

# Ask the agent a question about a stock
response = finance_agent.run("Should I buy Tesla stocks?")
print(response)
```

### Stock Sentiment Analysis with Visual Output

```python
# Run the interactive CLI tool
python stockSentiment.py
```

This will launch a terminal-based interactive tool with ASCII visualizations.

### Hedge Fund Growth Stock Analysis

```python
from searchRevenueGrowth import run_hedge_fund_growth_analysis

# Find high-growth stocks purchased by hedge funds
analysis = run_hedge_fund_growth_analysis(min_growth=30, years=3)
print(analysis)

# Or run it interactively
python searchRevenueGrowth.py
```

## Tool Descriptions

### 1. financeTest.py
Basic stock analysis tools including:
- `get_stock_data`: Fetches historical data for a stock
- `compare_stocks`: Compares multiple stocks over a specified time period
- `calculate_moving_averages`: Calculates technical indicators

### 2. stockSentiment.py
Enhanced version of financeTest with:
- Terminal-based ASCII visualizations
- Progress bars and colored output
- Portfolio stress testing via `simulate_market_scenarios`

### 3. searchRevenueGrowth.py
Tools for finding high-growth stocks:
- `analyze_revenue_growth`: Analyzes company revenue growth using Polygon.io API
- `search_hedge_fund_holdings`: Finds recent hedge fund stock purchases

## Example Scenarios

1. **Individual Stock Analysis**
   - "Should I buy Tesla stock?"
   - "What do the technical indicators say about AAPL?"

2. **Portfolio Comparison**
   - "Compare AAPL, MSFT, and GOOGL over the past year"
   - "Which tech stock performed best this year?"

3. **Recession Preparation**
   - "How would my portfolio of AAPL, JPM, and XOM perform in a recession?"
   - "What hedges should I add to protect against inflation?"

4. **Growth Stock Discovery**
   - "Find high-growth stocks that Renaissance Technologies has purchased"
   - "Which hedge fund purchases have 30%+ revenue growth?"

## Limitations

- The analysis is based on historical data and may not predict future performance
- API rate limits apply to data fetching operations
- Market simulations are simplified models and not financial advice

## License

This project is for educational purposes only. Always consult with a financial advisor before making investment decisions.

## Acknowledgments

This project uses:
- Claude 3.5 Sonnet by Anthropic for AI analysis
- yfinance for stock data retrieval
- smolagents for agent framework
- Polygon.io for financial data
