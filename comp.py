import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CORRECTED ALGO VS PLAIN INVESTMENT COMPARISON
# =============================================================================

class CorrectedAlgoVsPlainComparison:
    """Compare algorithmic trading results with plain buy-and-hold investment - NO COMPOUNDING"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.commission_percent = 0.05  # Same as algo
        self.slippage_percent = 0.02    # Same as algo
    
    def load_and_compare_corrected(self, excel_results_file: str, stock_data_folder: str, 
                                 output_file: str = None) -> pd.DataFrame:
        """
        Load ACTUAL algorithmic results (no new calculations) and compare with simple plain investment
        """
        print("CORRECTED ALGO vs PLAIN INVESTMENT COMPARISON")
        print("="*60)
        print("✓ Using ACTUAL algo results from midcap strategy file")
        print("✓ Calculating SIMPLE plain investment returns (no compounding)")
        
        # Load algorithmic results - USE AS-IS, NO NEW CALCULATIONS
        try:
            algo_results_df = pd.read_excel(excel_results_file, sheet_name='All_Results')
            print(f"✓ Loaded algorithmic results: {len(algo_results_df)} records")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return pd.DataFrame()
        
        # Calculate plain investment returns for unique stocks only (not per timeframe)
        unique_stocks = algo_results_df['symbol'].unique()
        stock_plain_returns = {}
        
        print(f"\nCalculating plain investment returns for {len(unique_stocks)} unique stocks...")
        
        for symbol in unique_stocks:
            plain_data = self._calculate_simple_plain_investment(symbol, stock_data_folder)
            stock_plain_returns[symbol] = plain_data
            
            if plain_data['plain_return_percent'] != 0:
                print(f"✓ {symbol}: {plain_data['plain_return_percent']:.2f}% return over {plain_data['plain_days_held']} days")
        
        # Add plain investment data to each row (same for all timeframes of same stock)
        for idx, row in algo_results_df.iterrows():
            symbol = row['symbol']
            plain_data = stock_plain_returns.get(symbol, self._empty_plain_data())
            
            # Add plain investment columns
            for key, value in plain_data.items():
                algo_results_df.loc[idx, key] = value
        
        # Calculate comparison metrics using ACTUAL algo results
        algo_results_df = self._calculate_corrected_comparison_metrics(algo_results_df)
        
        # Save corrected results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"corrected_algo_vs_plain_{timestamp}.xlsx"
        
        self._save_corrected_results(algo_results_df, output_file)
        self._display_corrected_summary(algo_results_df)
        
        return algo_results_df
    
    def _calculate_simple_plain_investment(self, symbol: str, stock_data_folder: str) -> Dict:
        """Calculate SIMPLE plain buy-and-hold investment returns - NO COMPOUNDING"""
        try:
            # Find stock file
            csv_files = [f for f in os.listdir(stock_data_folder) if f.endswith('.csv')]
            stock_file = None
            
            for csv_file in csv_files:
                if symbol.lower() in csv_file.lower():
                    stock_file = os.path.join(stock_data_folder, csv_file)
                    break
            
            if stock_file is None:
                print(f"  ⚠️  Stock file not found for {symbol}")
                return self._empty_plain_data()
            
            # Load and filter to same 6-month period
            df = pd.read_csv(stock_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to past 6 months (same as algo period)
            six_months_ago = datetime.now() - timedelta(days=180)
            df_filtered = df[df['date'] >= six_months_ago].copy()
            
            if df_filtered.empty:
                return self._empty_plain_data()
            
            df_filtered = df_filtered.sort_values('date')
            
            # SIMPLE CALCULATION: Buy at first price, sell at last price
            buy_price = df_filtered.iloc[0]['close']
            sell_price = df_filtered.iloc[-1]['close']
            
            # Use full initial capital (same as algo)
            investment_amount = self.initial_capital
            
            # Calculate costs
            buy_commission = investment_amount * (self.commission_percent / 100)
            buy_slippage = investment_amount * (self.slippage_percent / 100)
            net_investment = investment_amount - buy_commission - buy_slippage
            
            # Quantity purchased
            quantity = int(net_investment / buy_price)
            actual_cost = quantity * buy_price
            
            # Sell proceeds
            gross_proceeds = quantity * sell_price
            sell_commission = gross_proceeds * (self.commission_percent / 100)
            sell_slippage = gross_proceeds * (self.slippage_percent / 100)
            net_proceeds = gross_proceeds - sell_commission - sell_slippage
            
            # Simple P&L and return calculation
            total_costs = buy_commission + buy_slippage + sell_commission + sell_slippage
            net_pnl = net_proceeds - investment_amount
            
            # SIMPLE return percentage - NO COMPOUNDING
            plain_return = (net_pnl / investment_amount) * 100
            
            # Days held
            start_date = df_filtered.iloc[0]['date']
            end_date = df_filtered.iloc[-1]['date']
            days_held = (end_date - start_date).days
            
            return {
                'plain_investment_amount': investment_amount,
                'plain_buy_price': buy_price,
                'plain_sell_price': sell_price,
                'plain_quantity': quantity,
                'plain_net_pnl': net_pnl,
                'plain_return_percent': plain_return,
                'plain_total_costs': total_costs,
                'plain_days_held': days_held,
                'plain_margin_used': actual_cost,
                'plain_start_date': start_date.strftime('%Y-%m-%d'),
                'plain_end_date': end_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            return self._empty_plain_data()
    
    def _empty_plain_data(self) -> Dict:
        """Return empty plain investment data"""
        return {
            'plain_investment_amount': 0,
            'plain_buy_price': 0,
            'plain_sell_price': 0,
            'plain_quantity': 0,
            'plain_net_pnl': 0,
            'plain_return_percent': 0,
            'plain_total_costs': 0,
            'plain_days_held': 0,
            'plain_margin_used': 0,
            'plain_start_date': 'N/A',
            'plain_end_date': 'N/A'
        }
    
    def _calculate_corrected_comparison_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comparison using ACTUAL algo results vs simple plain investment"""
        print(f"\nCalculating corrected comparison metrics...")
        
        # Use ACTUAL algo results (no recalculation)
        # df already has: total_pnl, simple_return, avg_margin from midcap strategy
        
        # Comparison calculations
        df['pnl_difference'] = df['total_pnl'] - df['plain_net_pnl']
        df['return_difference'] = df['simple_return'] - df['plain_return_percent']
        df['margin_difference'] = df['avg_margin'] - df['plain_margin_used']
        
        # Determine winner based on actual P&L
        def determine_winner(row):
            if row['total_pnl'] > row['plain_net_pnl']:
                return 'Algorithmic Strategy'
            elif row['plain_net_pnl'] > row['total_pnl']:
                return 'Plain Investment'
            else:
                return 'Tie'
        
        df['performance_winner'] = df.apply(determine_winner, axis=1)
        
        # Performance edge calculation
        df['algo_performance_edge'] = np.where(
            df['plain_net_pnl'] != 0,
            ((df['total_pnl'] - df['plain_net_pnl']) / abs(df['plain_net_pnl']) * 100),
            0
        )
        
        return df
    
    def _save_corrected_results(self, df: pd.DataFrame, output_file: str):
        """Save corrected comparison results"""
        print(f"\nSaving corrected results to {output_file}...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main comparison with key columns
            comparison_cols = [
                'symbol', 'timeframe', 'total_trades', 'win_rate', 
                'total_pnl', 'simple_return', 'avg_margin',
                'plain_net_pnl', 'plain_return_percent', 'plain_margin_used',
                'pnl_difference', 'return_difference', 'margin_difference',
                'performance_winner', 'algo_performance_edge'
            ]
            
            df[comparison_cols].to_excel(writer, sheet_name='Algo_vs_Plain_Comparison', index=False)
            
            # Winner summary
            winner_summary = df.groupby('performance_winner').agg({
                'symbol': 'count',
                'total_pnl': 'sum',
                'plain_net_pnl': 'sum',
                'pnl_difference': 'sum',
                'return_difference': 'mean',
                'simple_return': 'mean',
                'plain_return_percent': 'mean'
            }).round(2)
            winner_summary.columns = ['Count', 'Total_Algo_PnL', 'Total_Plain_PnL', 
                                    'Total_PnL_Diff', 'Avg_Return_Diff', 
                                    'Avg_Algo_Return', 'Avg_Plain_Return']
            winner_summary.to_excel(writer, sheet_name='Winner_Summary')
            
            # Stock-wise comparison (one row per stock - best timeframe)
            stock_comparison = df.loc[df.groupby('symbol')['total_pnl'].idxmax()]
            stock_comparison[['symbol', 'timeframe', 'total_pnl', 'simple_return',
                           'plain_net_pnl', 'plain_return_percent', 'pnl_difference',
                           'return_difference', 'performance_winner']].to_excel(
                writer, sheet_name='Stock_Comparison', index=False)
        
        print(f"✓ Corrected results saved")
    
    def _display_corrected_summary(self, df: pd.DataFrame):
        """Display corrected comparison summary"""
        successful_tests = df[df['status'] == 'Success']
        
        if successful_tests.empty:
            print("No successful tests to compare")
            return
        
        print(f"\n{'='*70}")
        print("CORRECTED ALGO vs PLAIN INVESTMENT SUMMARY")
        print(f"{'='*70}")
        
        # Overall performance using ACTUAL results
        total_algo_pnl = successful_tests['total_pnl'].sum()
        total_plain_pnl = successful_tests['plain_net_pnl'].sum()
        net_difference = total_algo_pnl - total_plain_pnl
        
        print(f"\nOVERALL PERFORMANCE (CORRECTED):")
        print(f"Total Algorithmic P&L: ₹{total_algo_pnl:,.2f}")
        print(f"Total Plain Investment P&L: ₹{total_plain_pnl:,.2f}")
        print(f"Net Difference: ₹{net_difference:,.2f}")
        print(f"Overall Winner: {'ALGORITHMIC STRATEGY' if net_difference > 0 else 'PLAIN INVESTMENT'}")
        
        # Performance breakdown
        winner_counts = successful_tests['performance_winner'].value_counts()
        total_tests = len(successful_tests)
        
        print(f"\nPERFORMANCE BREAKDOWN:")
        for winner, count in winner_counts.items():
            percentage = (count / total_tests) * 100
            print(f"{winner}: {count} tests ({percentage:.1f}%)")
        
        # Average returns (corrected)
        avg_algo_return = successful_tests['simple_return'].mean()
        avg_plain_return = successful_tests['plain_return_percent'].mean()
        
        print(f"\nCORRECTED AVERAGE RETURNS:")
        print(f"Average Algo Return: {avg_algo_return:.2f}%")
        print(f"Average Plain Return: {avg_plain_return:.2f}%")
        print(f"Average Return Advantage: {(avg_algo_return - avg_plain_return):.2f}%")
        
        # Best performers
        best_algo = successful_tests.loc[successful_tests['pnl_difference'].idxmax()]
        worst_algo = successful_tests.loc[successful_tests['pnl_difference'].idxmin()]
        
        print(f"\nBEST ALGO PERFORMANCE:")
        print(f"{best_algo['symbol']} ({best_algo['timeframe']}min): " +
              f"₹{best_algo['pnl_difference']:,.2f} advantage")
        
        print(f"\nWORST ALGO PERFORMANCE:")
        print(f"{worst_algo['symbol']} ({worst_algo['timeframe']}min): " +
              f"₹{worst_algo['pnl_difference']:,.2f} vs plain investment")
        
        # Realistic timeframe analysis
        print(f"\nTIMEFRAME ANALYSIS (CORRECTED):")
        for timeframe in [15, 30, 60]:
            tf_data = successful_tests[successful_tests['timeframe'] == timeframe]
            if not tf_data.empty:
                algo_wins = len(tf_data[tf_data['performance_winner'] == 'Algorithmic Strategy'])
                win_rate = (algo_wins / len(tf_data)) * 100
                avg_return_diff = tf_data['return_difference'].mean()
                avg_pnl_diff = tf_data['pnl_difference'].mean()
                
                print(f"{timeframe}min: {algo_wins}/{len(tf_data)} wins ({win_rate:.1f}%), " +
                      f"Avg return edge: {avg_return_diff:.2f}%, Avg P&L edge: ₹{avg_pnl_diff:,.0f}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_corrected_algo_vs_plain_comparison(excel_results_file: str, stock_data_folder: str, 
                                         output_file: str = None):
    """
    Run CORRECTED algorithmic vs plain investment comparison
    
    Args:
        excel_results_file: Path to Excel file with ACTUAL algorithmic results
        stock_data_folder: Path to folder containing stock CSV data files  
        output_file: Optional output Excel file name
    """
    comparator = CorrectedAlgoVsPlainComparison()
    results_df = comparator.load_and_compare_corrected(excel_results_file, stock_data_folder, output_file)
    
    return results_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Execute corrected comparison
    excel_file = "midcap_strategy_results.xlsx"  # Your actual midcap results
    data_folder = "Midcap Data"  # Your stock CSV files
    output_file = "corrected_algo_vs_plain_comparison.xlsx"
    
    print("STARTING CORRECTED ALGO vs PLAIN COMPARISON")
    print("="*55)
    print("✓ Using ACTUAL algorithmic results (no recalculation)")
    print("✓ Calculating SIMPLE plain investment returns")
    print("✓ No compounding or unrealistic calculations")
    
    try:
        results = run_corrected_algo_vs_plain_comparison(excel_file, data_folder, output_file)
        print(f"\n✅ Corrected comparison completed!")
        print(f"📊 Realistic results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

    print(f"\n{'='*55}")
    print("CORRECTED COMPARISON COMPLETE")
    print("="*55)
