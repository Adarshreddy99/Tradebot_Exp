from kiteconnect import KiteConnect
import datetime
import pandas as pd
import time

# ---- CONFIGURE YOUR API CREDENTIALS ----
api_key = "2efvj0f8goaio9dm"
access_token = "7mlMOSyrLw03VHG01TEZW9zw2VI0qYrS"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# ---- Latest verified Nifty 50 symbols for reference ----
nifty_50_stocks = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
]

nifty_midcap100 = [
    "DIXON", "SOLARINDS", "SUZLON", "HINDZINC", "BSE", "HDFCAMC", "FEDERALBNK", 
    "BANDHANBNK", "IDFCFIRSTB", "YESBANK", "KPITTECH", "PERSISTENT", "COFORGE", 
    "TATAELXSI", "TATATECH", "MAZDOCK", "COCHINSHIP", "APOLLOTYRE", "MRF", "EXIDEIND"]

nifty_smallcap100 = [
    "IDBI", "MCX", "CDSL", "ANGELONE", "BANDHANBNK", "AFFLE",
    "LAURUSLABS", "GRSE", "KAYNES", "DELHIVERY", "NH", "ASTERDM", 
    "LALPATHLAB", "GODIGIT", "SHYAMMETL", "RADICO", "PEL", "NBCC", 
    "IKS", "GODFRYPHLP"]


# ---- List of stocks you want to fetch
stock_list = nifty_smallcap100

# ---- Helper: Get instrument token
def get_instrument_token(symbol):
    try:
        ltp = kite.ltp([f"NSE:{symbol}"])
        return ltp[f"NSE:{symbol}"]["instrument_token"]
    except Exception as e:
        print(f"Error fetching instrument_token for {symbol}: {e}")
        return None

# ---- Fetch historical data in batches
def fetch_stock_data(symbol, from_date, to_date, interval="minute", chunk_days=30):
    token = get_instrument_token(symbol)
    if token is None:
        return []

    all_data = []
    chunk = datetime.timedelta(days=chunk_days)
    start = from_date

    print(f"Starting data pull for {symbol}...")
    while start < to_date:
        end = min(start + chunk, to_date)
        try:
            bars = kite.historical_data(token, start, end, interval=interval)
            if bars:
                all_data.extend(bars)
            print(f"{symbol}: Retrieved {len(bars)} records from {start.date()} to {end.date()}")
            time.sleep(0.5)  # respect API rate limits
        except Exception as e:
            print(f"Error retrieving data for {symbol} from {start} to {end}: {e}")
        start = end + datetime.timedelta(minutes=1)
    return all_data

# ---- Save each stock’s data to individual CSV files
def fetch_and_save_stocks_to_csv(stock_list, date_from, date_to):
    for idx, symbol in enumerate(stock_list, start=1):
        data = fetch_stock_data(symbol, date_from, date_to)
        if data:
            df = pd.DataFrame(data)
            # Remove timezone info from 'date' column if present for compatibility
            if 'date' in df.columns and pd.api.types.is_datetime64tz_dtype(df['date']):
                df['date'] = df['date'].dt.tz_localize(None)
            csv_filename = f"{symbol}_1min_data.csv"
            try:
                df.to_csv(csv_filename, index=False)
                print(f"[{idx}/{len(stock_list)}] Saved {symbol} data to {csv_filename}.")
            except Exception as e:
                print(f"Error writing CSV for {symbol}: {e}")
            del df  # free memory
        else:
            print(f"[{idx}/{len(stock_list)}] No data for {symbol}")

# ---- Date range (customize as needed)
from_date = datetime.datetime(2020, 8, 7, 9, 15)
to_date = datetime.datetime(2025, 8, 7, 15, 30)

# ---- Run for selected stocks
fetch_and_save_stocks_to_csv(
    stock_list,
    date_from=from_date,
    date_to=to_date
)
