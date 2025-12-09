import pandas as pd
import numpy as np
import os

# ==========================================
# FILE SETUP
# ==========================================
PRICE_FILE = 'prices_panel.csv'
FUNDAMENTALS_FILE = 'fundamentals_income.csv'
OUTPUT_FILE = 'feature_panel_daily.csv'

def clean_col_names(df):
    """Standardize column names to snake_case."""
    df.columns = (df.columns.str.lower()
                  .str.replace(' ', '_')
                  .str.replace('.', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('/', '_', regex=False)
                  .str.replace('&', '', regex=False))
    return df

def prepare_data_panel():
    print(f"Loading data from {PRICE_FILE} and {FUNDAMENTALS_FILE}...")
    
    # --- 1. Load and Prep Data ---
    prices = pd.read_csv(PRICE_FILE)
    prices = clean_col_names(prices)
    prices['date'] = pd.to_datetime(prices['date'])
    # CRITICAL: Sort by DATE for the merge_asof lookup
    prices = prices.sort_values('date')

    earnings = pd.read_csv(FUNDAMENTALS_FILE)
    earnings = clean_col_names(earnings)
    if 'publish_date' in earnings.columns:
        earnings = earnings.rename(columns={'publish_date': 'date'})
    earnings['date'] = pd.to_datetime(earnings['date'])
    
    # --- 2. Calculate Fundamental Ratios ---
    share_col = 'shares_diluted' if 'shares_diluted' in earnings.columns else 'shares_basic'
    
    earnings['eps_raw'] = earnings['net_income'] / earnings[share_col].replace(0, np.nan)
    earnings['gross_margin'] = earnings['gross_profit'] / earnings['revenue'].replace(0, np.nan)
    
    fund_cols = ['ticker', 'date', 'eps_raw', 'gross_margin']
    fund_subset = earnings[fund_cols].dropna(subset=['date', 'ticker'])
    fund_subset = fund_subset.sort_values('date')
    
    # --- 3. MERGE Fundamentals (FFILL via merge_asof) ---
    # Merges the most recent fundamental data (backward direction) onto every trading day.
    print("Merging fundamentals (Forward Fill)...")
    df_merged = pd.merge_asof(prices, fund_subset,
                              on='date',
                              by='ticker',
                              direction='backward',
                              suffixes=('_price', '_fund'))
    
    # --- 4. PRICE FEATURE ENGINEERING ---
    print("Calculating rolling price features and targets...")
    # Re-sort by Ticker/Date for rolling window calculations
    df = df_merged.copy().sort_values(['ticker', 'date'])
    
    # Rolling Windows
    windows = {'1d': 1, '1w': 5, '1m': 21, '3m': 63}
    horizons = {'1w': 5, '1m': 21, '3m': 63}
    
    for label, days in windows.items():
        # Momentum & Volume
        df[f'acc_ret_{label}'] = df.groupby('ticker')['adj_close'].pct_change(days)
        df[f'vol_{label}'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(days).mean())
        
        if days > 1:
            # Volatility, Skew, Max Drawdown, Dist from Mean
            df[f'vola_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).std()) * np.sqrt(252)
            df[f'skew_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).skew())
            
            roll_max = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).max())
            df[f'max_dd_{label}'] = (df['adj_close'] / roll_max) - 1
            roll_mean = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).mean())
            df[f'dist_mean_{label}'] = (df['adj_close'] / roll_mean) - 1

    # 4b. Forward Returns (Targets)
    for label, days in horizons.items():
        df[f'fwd_ret_{label}'] = df.groupby('ticker')['adj_close'].shift(-days) / df['adj_close'] - 1

    # --- 5. Final Ratios & Cleaning ---
    df['earnings_yield'] = df['eps_raw'] / df['adj_close']
    df['pe_ratio'] = df['adj_close'] / df['eps_raw'].replace(0, np.nan)
    
    # Critical Fix: Convert INF/-INF to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

# --- 6. EXECUTION AND SAVE ---
if __name__ == "__main__":
    df_panel = prepare_data_panel()
    
    # Filter only useful columns before saving
    cols_to_keep = ['date', 'ticker', 'adj_close', 'eps_raw', 'gross_margin', 
                    'earnings_yield', 'pe_ratio']
    
    # Add all calculated features and targets
    calculated_cols = [col for col in df_panel.columns if any(x in col for x in ['acc_ret_', 'vol_', 'vola_', 'skew_', 'max_dd_', 'dist_mean_', 'fwd_ret_'])]
    
    final_cols = list(set(cols_to_keep + calculated_cols))
    df_output = df_panel[final_cols].copy()
    
    df_output = df_output.sort_values(['ticker', 'date']).dropna(subset=['adj_close'])
    
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSuccessfully created panel with {len(df_output)} daily observations.")
    print(f"File saved as {OUTPUT_FILE}")