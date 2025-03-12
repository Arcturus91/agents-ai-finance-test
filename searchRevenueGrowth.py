import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable is not set")

# Base URL for Polygon.io API
BASE_URL = "https://api.polygon.io"

# Function to test API connection and get stock data
def test_polygon_api():
    # Calculate dates for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Format dates as YYYY-MM-DD
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Test 1: Get ticker details
    ticker = "AAPL"
    ticker_url = f"{BASE_URL}/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
    
    print(f"Testing API connection for ticker details: {ticker}")
    ticker_response = requests.get(ticker_url)
    
    if ticker_response.status_code == 200:
        ticker_data = ticker_response.json()
        print(f"✅ Successfully retrieved ticker details for {ticker}")
        print(f"Company name: {ticker_data['results']['name']}")
        print(f"Market cap: {ticker_data['results'].get('market_cap', 'N/A')}")
        print(f"Primary exchange: {ticker_data['results'].get('primary_exchange', 'N/A')}")
        print()
    else:
        print(f"❌ Failed to retrieve ticker details. Status code: {ticker_response.status_code}")
        print(f"Response: {ticker_response.text}")
        print()
    
    # Test 2: Get daily close prices for the last 30 days
    aggs_url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date_str}/{end_date_str}?apiKey={POLYGON_API_KEY}"
    
    print(f"Testing API connection for price data: {ticker} from {start_date_str} to {end_date_str}")
    aggs_response = requests.get(aggs_url)
    
    if aggs_response.status_code == 200:
        aggs_data = aggs_response.json()
        if aggs_data['results'] and len(aggs_data['results']) > 0:
            print(f"✅ Successfully retrieved price data: {len(aggs_data['results'])} days of data")
            
            # Show the most recent 5 days
            recent_data = aggs_data['results'][-5:]
            for day in recent_data:
                date = datetime.fromtimestamp(day['t'] / 1000).strftime('%Y-%m-%d')
                print(f"Date: {date}, Close: ${day['c']}, Volume: {day['v']:,}")
        else:
            print("✅ API connection successful but no results returned")
    else:
        print(f"❌ Failed to retrieve price data. Status code: {aggs_response.status_code}")
        print(f"Response: {aggs_response.text}")
    
    # Test 3: Get company financials
    financials_url = f"{BASE_URL}/vX/reference/financials?ticker={ticker}&limit=1&apiKey={POLYGON_API_KEY}"
    
    print(f"\nTesting API connection for financial data: {ticker}")
    financials_response = requests.get(financials_url)
    
    if financials_response.status_code == 200:
        financials_data = financials_response.json()
        if financials_data['results'] and len(financials_data['results']) > 0:
            print(f"✅ Successfully retrieved financial data")
            
            # Get the most recent financial data
            recent_financial = financials_data['results'][0]
            print(f"Filing date: {recent_financial.get('filing_date', 'N/A')}")
            print(f"Period end date: {recent_financial.get('end_date', 'N/A')}")
            
            # Extract revenue if available
            income_stmt = recent_financial.get('financials', {}).get('income_statement', {})
            revenue = income_stmt.get('revenues', {}).get('value', 'N/A')
            net_income = income_stmt.get('net_income_loss', {}).get('value', 'N/A')
            
            print(f"Revenue: ${revenue:,}" if isinstance(revenue, (int, float)) else f"Revenue: {revenue}")
            print(f"Net Income: ${net_income:,}" if isinstance(net_income, (int, float)) else f"Net Income: {net_income}")
        else:
            print("✅ API connection successful but no financial results returned")
    else:
        print(f"❌ Failed to retrieve financial data. Status code: {financials_response.status_code}")
        print(f"Response: {financials_response.text}")

if __name__ == "__main__":
    print("Starting Polygon.io API Test...")
    test_polygon_api()
    print("\nAPI test completed.")