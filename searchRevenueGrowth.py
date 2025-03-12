import os
import requests
import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any
from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, LiteLLMModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable is not set")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

# Base URL for Polygon.io API
BASE_URL = "https://api.polygon.io"

# Helper functions for the revenue growth tool
def get_financials(ticker, limit=20, timeframe="annual"):
    """Get financial data for a ticker"""
    ticker = ticker.upper()
    url = f"{BASE_URL}/vX/reference/financials?ticker={ticker}&limit={limit}&timeframe={timeframe}&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return None
        
    return response.json()

def extract_revenue_data(financials_json):
    """Extract revenue data from financials JSON"""
    if not financials_json or 'results' not in financials_json or not financials_json['results']:
        return None
        
    revenue_data = []
    
    for result in financials_json['results']:
        if 'financials' in result and 'income_statement' in result['financials']:
            income_stmt = result['financials']['income_statement']
            
            # Try to get revenue (might be under different keys)
            revenue = None
            for key in ['revenues', 'revenue', 'total_revenues', 'total_revenue']:
                if key in income_stmt and 'value' in income_stmt[key]:
                    revenue = income_stmt[key]['value']
                    break
            
            if revenue is not None:
                end_date = result.get('end_date')
                filing_date = result.get('filing_date')
                
                if end_date:
                    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    revenue_data.append({
                        'end_date': end_date,
                        'end_date_dt': end_date_dt,
                        'filing_date': filing_date,
                        'revenue': revenue,
                        'fiscal_year': end_date_dt.year,
                        'fiscal_period': result.get('fiscal_period', '')
                    })
    
    if not revenue_data:
        return None
        
    # Convert to DataFrame and sort by date
    df = pd.DataFrame(revenue_data)
    df = df.sort_values('end_date_dt')
    
    return df

def calculate_growth_rates(revenue_df):
    """Calculate year-over-year growth rates"""
    if revenue_df is None or revenue_df.empty or len(revenue_df) < 2:
        return None
        
    # Make a copy to avoid modifying the original
    df = revenue_df.copy()
    
    # Calculate year-over-year growth
    df['previous_revenue'] = df['revenue'].shift(1)
    df['growth_rate'] = ((df['revenue'] - df['previous_revenue']) / df['previous_revenue']) * 100
    
    # Format growth rate to 2 decimal places
    df['growth_rate'] = df['growth_rate'].round(2)
    
    # Drop rows with NaN growth rate
    df = df.dropna(subset=['growth_rate'])
    
    return df

@tool
def analyze_revenue_growth(ticker: str, years: int = 3) -> Dict:
    """Analyzes the revenue growth of a company over multiple years using Polygon.io API
    
    Args:
        ticker: The stock ticker symbol to analyze (e.g., AAPL)
        years: Number of years to analyze
    
    Returns:
        Dictionary containing growth analysis results
    """
    if not ticker:
        return {"error": "Ticker symbol is required"}
        
    # Fetch more data than needed to ensure we have enough
    limit = years * 2
    
    # Get financial data
    financials = get_financials(ticker, limit=limit)
    if not financials:
        return {
            "ticker": ticker,
            "success": False,
            "error": "Failed to fetch financial data"
        }
    
    # Extract revenue data
    revenue_df = extract_revenue_data(financials)
    if revenue_df is None or revenue_df.empty:
        return {
            "ticker": ticker,
            "success": False,
            "error": "No revenue data found"
        }
        
    # Calculate growth rates
    growth_df = calculate_growth_rates(revenue_df)
    if growth_df is None or growth_df.empty:
        return {
            "ticker": ticker,
            "success": False,
            "error": "Could not calculate growth rates"
        }
        
    # Get the most recent years data
    recent_growth = growth_df.tail(years)
    
    # Calculate average growth over the period
    avg_growth = recent_growth['growth_rate'].mean()
    
    # Check if growth is above threshold (30%)
    above_threshold = avg_growth >= 30
    
    # Prepare year-by-year growth data
    yearly_growth = []
    for _, row in recent_growth.iterrows():
        yearly_growth.append({
            "year": row['end_date'],
            "revenue": float(row['revenue']),
            "growth_rate": float(row['growth_rate'])
        })
        
    return {
        "ticker": ticker,
        "success": True,
        "years_analyzed": len(recent_growth),
        "avg_growth_rate": round(float(avg_growth), 2),
        "above_threshold": above_threshold,
        "yearly_growth": yearly_growth
    }

def extract_tickers(text):
    """Extract ticker symbols from text"""
    # Look for ticker patterns: uppercase letters often in parentheses
    ticker_patterns = [
        r'\(([A-Z]{1,5})\)',  # Tickers in parentheses like (AAPL)
        r'NYSE:\s*([A-Z]{1,5})', # NYSE: AAPL
        r'NASDAQ:\s*([A-Z]{1,5})',  # NASDAQ: AAPL
        r'\b([A-Z]{2,5})\b'  # Standalone uppercase words like AAPL
    ]
    
    potential_tickers = []
    for pattern in ticker_patterns:
        matches = re.findall(pattern, text)
        potential_tickers.extend(matches)
        
    # Filter out common words that might be mistaken as tickers
    common_words = {'CEO', 'CFO', 'COO', 'CTO', 'THE', 'FOR', 'AND', 'FROM', 'LLC', 'INC', 'LP', 'ETF'}
    filtered_tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
    
    # Return unique tickers
    return list(set(filtered_tickers))

@tool
def search_hedge_fund_holdings(fund_name: Optional[str] = None) -> Dict:
    """Searches for recent hedge fund stock purchases and extracts ticker symbols
    
    Args:
        fund_name: Name of the hedge fund to focus on (optional)
    
    Returns:
        Dictionary containing extracted tickers and search results
    """
    # Create a new search tool instance
    search_tool = DuckDuckGoSearchTool()
    
    if fund_name:
        query = f"latest stock purchases by {fund_name} hedge fund"
    else:
        query = "latest hedge fund stock purchases this quarter"
        
    search_results = search_tool.search(query)
    
    # Extract potential ticker symbols from search results
    tickers = []
    for result in search_results:
        extracted = extract_tickers(result['body'])
        tickers.extend(extracted)
        
    # Return unique tickers
    return {"tickers": list(set(tickers)), "source": search_results}

def run_hedge_fund_growth_analysis(min_growth=30, years=3, fund_name=None):
    """
    Run the hedge fund growth analysis
    
    Args:
        min_growth (float): Minimum growth rate threshold
        years (int): Number of years to analyze
        fund_name (str, optional): Name of hedge fund to focus on
        
    Returns:
        str: Analysis results
    """
    # Initialize Claude model
    model = LiteLLMModel(
        model_id="anthropic/claude-3-5-sonnet-latest",
        temperature=0.2,
        api_key=ANTHROPIC_API_KEY
    )
    
    # Create a code agent with the tools
    agent = CodeAgent(tools=[analyze_revenue_growth, search_hedge_fund_holdings], model=model)
    
    prompt = f"""
    You are a financial analyst specializing in finding high-growth companies. Follow these steps exactly:
    
    1. Use the search_hedge_fund_holdings tool to find recent stock purchases by hedge funds
       {f'Focus on {fund_name}' if fund_name else 'across major hedge funds'}
    
    2. For each ticker symbol found, use the analyze_revenue_growth tool to analyze its revenue growth over the past {years} years
    
    3. Filter for companies with average revenue growth of at least {min_growth}% over the period
    
    4. Provide a detailed analysis of each qualifying company including:
       - Ticker symbol
       - Average revenue growth rate
       - Year-by-year growth
       - Brief description of the company
    
    Format the results clearly with markdown for readability. Focus only on companies that meet the {min_growth}% growth threshold.
    """
    
    return agent.run(prompt)


if __name__ == "__main__":
    # Interactive mode
    print("=== Hedge Fund Growth Stock Analyzer ===")
    print("This tool finds hedge fund investments with strong revenue growth.")
    
    while True:
        print("\nOptions:")
        print("1. Analyze stocks from all major hedge funds")
        print("2. Analyze stocks from a specific hedge fund")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3':
            print("Exiting program. Goodbye!")
            break
            
        # Get parameters
        try:
            growth_threshold = input("Enter minimum growth threshold % (default: 30): ").strip()
            min_growth = float(growth_threshold) if growth_threshold else 30
            
            years_input = input("Enter number of years to analyze (default: 3): ").strip()
            years = int(years_input) if years_input else 3
        except ValueError:
            print("Invalid input. Using defaults (30% growth over 3 years).")
            min_growth = 30
            years = 3
            
        fund_name = None
        if choice == '2':
            fund_name = input("Enter hedge fund name (e.g., Renaissance Technologies): ").strip()
            
        # Run analysis
        print(f"\nSearching for companies with {min_growth}%+ revenue growth over {years} years...")
        print("This may take a few minutes. Please wait...\n")
        
        result = run_hedge_fund_growth_analysis(min_growth, years, fund_name)
        print(result)