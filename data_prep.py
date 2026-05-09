import pandas as pd
import numpy as np
import os

def compute_indicators_with_target(file_path, output_path):
    """
    Compute technical indicators and target labels for trading data.
    
    Fixed parameters:
    - Lookahead: 20 minutes
    - Penalty weight: 1.5 (drops weighted 1.5x more than gains)
    - Peak proximity threshold: 15 minutes
    """
    # Load and prepare data
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS COMPUTATION
    # ═══════════════════════════════════════════════════════════════════
    
    # 1. ATR (Average True Range) - 21 periods
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_21'] = true_range.rolling(window=21, min_periods=21).mean()
    
    # 2. Exponential Moving Averages
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # 3. Simple Moving Averages
    df['SMA_21'] = df['close'].rolling(window=21, min_periods=21).mean()
    df['SMA_42'] = df['close'].rolling(window=42, min_periods=42).mean()
    
    # 4. RSI (Relative Strength Index) - 14 periods
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 5. Bollinger Bands - 21 periods, 2 standard deviations
    rolling_mean = df['close'].rolling(21).mean()
    rolling_std = df['close'].rolling(21).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    # 6. OBV (On-Balance Volume)
    obv_values = [0]
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i-1, 'close']:
            obv_values.append(obv_values[-1] + df.loc[i, 'volume'])
        elif df.loc[i, 'close'] < df.loc[i-1, 'close']:
            obv_values.append(obv_values[-1] - df.loc[i, 'volume'])
        else:
            obv_values.append(obv_values[-1])
    df['OBV'] = obv_values
    
    # 7. CMF (Chaikin Money Flow) - 21 periods
    money_flow_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_mult = money_flow_mult.replace([np.inf, -np.inf], 0).fillna(0)
    money_flow_volume = money_flow_mult * df['volume']
    df['CMF_21'] = (money_flow_volume.rolling(window=21).sum() / 
                    df['volume'].rolling(window=21).sum())
    
    # 8. VWAP (Volume Weighted Average Price)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED TARGET LABELING LOGIC
    # ═══════════════════════════════════════════════════════════════════
    
    LOOKAHEAD_MINUTES = 180
    PEAK_DOWNFALL_THRESHOLD = 15  # Minutes of continuous decline from peak
    MIN_INCREASE_THRESHOLD = 1.5  # Minimum % increase to consider significant
    
    targets = []
    close_prices = df['close'].values
    n = len(df)
    
    print(f"Processing {n} data points with enhanced {LOOKAHEAD_MINUTES}-minute lookahead logic...")
    
    for i in range(n):
        # Default label
        label = "Not Buy"
        
        # Check if we have enough future data
        if i + LOOKAHEAD_MINUTES >= n:
            targets.append(label)
            continue
        
        # Get current price and future prices for next 20 minutes
        current_price = close_prices[i]
        future_prices = close_prices[i+1:i+LOOKAHEAD_MINUTES+1]  # Next 20 minutes
        
        # Calculate percentage changes from current price for each future minute
        percent_changes = ((future_prices - current_price) / current_price) * 100
        
        # Find the peak (highest price) in the lookahead window
        peak_index = np.argmax(future_prices)
        peak_price = future_prices[peak_index]
        peak_percent_change = ((peak_price - current_price) / current_price) * 100
        
        # Check for continuous downfall after peak (Peak Logic)
        continuous_downfall = False
        if peak_index < len(future_prices) - PEAK_DOWNFALL_THRESHOLD:
            # Check if there's continuous decline for PEAK_DOWNFALL_THRESHOLD minutes after peak
            post_peak_prices = future_prices[peak_index:peak_index + PEAK_DOWNFALL_THRESHOLD + 1]
            
            # Check if prices are continuously declining from peak
            downfall_count = 0
            for j in range(1, len(post_peak_prices)):
                if post_peak_prices[j] < post_peak_prices[j-1]:
                    downfall_count += 1
                else:
                    break
            
            if downfall_count >= PEAK_DOWNFALL_THRESHOLD:
                continuous_downfall = True
        
        # Final price at 20th minute
        final_price = future_prices[-1]
        final_percent_change = ((final_price - current_price) / current_price) * 100
        
        # Decision Logic:
        # 1. NOT BUY if there's continuous downfall after peak
        if continuous_downfall:
            label = "Not Buy"
        
        # 2. NOT BUY if final price decreased from current price
        elif final_percent_change <= 0:
            label = "Not Buy"
        
        # 3. NOT BUY if increase is not significant
        elif final_percent_change < MIN_INCREASE_THRESHOLD:
            label = "Not Buy"
        
        # 4. BUY CONDITIONS:
        # - Steady increase pattern OR
        # - Even if there are dips, final price is significantly higher than current
        else:
            # Check for overall positive trend
            positive_minutes = sum(1 for pct in percent_changes if pct > 0)
            total_minutes = len(percent_changes)
            positive_ratio = positive_minutes / total_minutes
            
            # Buy if:
            # - Final price is significantly higher AND
            # - Either majority of minutes are positive OR final gain is substantial
            if (final_percent_change >= MIN_INCREASE_THRESHOLD and
                (positive_ratio >= 0.6 or final_percent_change >= 1.0)):
                label = "Buy"
        
        targets.append(label)
    
    # Add target column to dataframe
    df['target'] = targets
    
    # Summary statistics
    buy_count = sum(1 for t in targets if t == "Buy")
    total_count = len(targets)
    buy_percentage = (buy_count / total_count) * 100
    
    print(f"Target Distribution:")
    print(f"  Buy: {buy_count} ({buy_percentage:.2f}%)")
    print(f"  Not Buy: {total_count - buy_count} ({100 - buy_percentage:.2f}%)")
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # List of stock symbols to process
    stock_list = [ 
    "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
]
    
    # Create output directory
    output_dir = "Nifty_Data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # Process each stock
    successful_files = 0
    failed_files = []
    
    print(f"Starting to process {len(stock_list)} stock files...")
    print("="*80)
    
    for stock in stock_list:
        # Input and output file paths
        input_file = f"{stock}_1min_data.csv"
        output_file = os.path.join(output_dir, f"{stock}_data.csv")
        
        print(f"\n🔄 Processing {stock}...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        
        try:
            df_result = compute_indicators_with_target(
                file_path=input_file,
                output_path=output_file
            )
            successful_files += 1
            
        except FileNotFoundError:
            print(f"❌ Error: Input file '{input_file}' not found.")
            failed_files.append(stock)
        except Exception as e:
            print(f"❌ Error processing {stock}: {str(e)}")
            failed_files.append(stock)
    
    # Final summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"✅ Successfully processed: {successful_files}/{len(stock_list)} files")
    print(f"📁 Output directory: {output_dir}")
    
    if failed_files:
        print(f"\n❌ Failed to process ({len(failed_files)} files):")
        for stock in failed_files:
            print(f"   - {stock}_1min_data.csv")
        print("\nPlease ensure these CSV files exist in the current directory.")
    else:
        print("\n🎉 All files processed successfully!")