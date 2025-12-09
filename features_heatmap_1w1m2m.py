import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING & SETUP
# ==========================================
PRICE_FILE = 'prices_panel.csv'
FUNDAMENTALS_FILE = 'fundamentals_income.csv'

def clean_col_names(df):
    df.columns = (df.columns.str.lower()
                  .str.replace(' ', '_')
                  .str.replace('.', '')
                  .str.replace(',', '')
                  .str.replace('(', '')
                  .str.replace(')', '')
                  .str.replace('/', '_')
                  .str.replace('&', ''))
    return df

def load_data():
    try:
        prices = pd.read_csv(PRICE_FILE)
        prices = clean_col_names(prices)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices['date'] = pd.to_datetime(prices['date'])
        
        earnings = pd.read_csv(FUNDAMENTALS_FILE)
        earnings = clean_col_names(earnings)
        earnings = earnings.loc[:, ~earnings.columns.duplicated()]
        
        if 'publish_date' in earnings.columns:
            earnings = earnings.rename(columns={'publish_date': 'date'})
        
        earnings['date'] = pd.to_datetime(earnings['date'])
        earnings = earnings.dropna(subset=['date']).copy()

        share_col = 'shares_diluted' if 'shares_diluted' in earnings.columns else 'shares_basic'
        
        if 'net_income' in earnings.columns and share_col in earnings.columns:
            earnings['eps'] = earnings['net_income'] / earnings[share_col].replace(0, np.nan)
        else:
            earnings['eps'] = np.nan

        if 'gross_profit' in earnings.columns and 'revenue' in earnings.columns:
            earnings['gross_margin'] = earnings['gross_profit'] / earnings['revenue'].replace(0, np.nan)
        else:
            earnings['gross_margin'] = np.nan

        return prices, earnings
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def calculate_features(price_df, fund_df):
    df = price_df.copy().sort_values(['ticker', 'date'])
    df = df.drop_duplicates(subset=['ticker', 'date'])
    
    windows = {'1d': 1, '1w': 5, '1m': 21, '2m': 42}
    
    for label, days in windows.items():
        # Accumulate Return
        df[f'acc_ret_{label}'] = df.groupby('ticker')['adj_close'].pct_change(days)
        
        # Accumulate Volume
        vol_sum = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(days).sum())
        df[f'acc_vol_{label}'] = np.log1p(vol_sum) 
        
        # Dist to Rolling Mean
        if label != '1d':
            roll_mean = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).mean())
            df[f'dist_mean_{label}'] = (df['adj_close'] / roll_mean) - 1

        # Volatility & Skew
        if days > 1:
            df[f'vol_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).std()) * np.sqrt(252)
            df[f'skew_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).skew())
        
        # Max Drawdown
        if label != '1d':
            roll_min = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).min())
            roll_max = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).max())
            df[f'maxdd_{label}'] = (roll_min / roll_max) - 1

    # Targets (Horizons)
    horizons = {'1w': 5, '1m': 21, '2m': 42}
    for label, days in horizons.items():
        df[f'fwd_ret_{label}'] = df.groupby('ticker')['adj_close'].shift(-days) / df['adj_close'] - 1
        
    # Merge Fundamentals
    cols_to_keep = ['ticker', 'date', 'eps', 'gross_margin']
    cols_to_keep = [c for c in cols_to_keep if c in fund_df.columns]
    fund_subset = fund_df[cols_to_keep].copy()
    
    merged = pd.merge(df, fund_subset, on=['ticker', 'date'], how='inner')
    
    # Binary Targets
    for label in horizons.keys():
        col_name = f'Target_Neg_{label}'
        merged[col_name] = (merged[f'fwd_ret_{label}'] < 0).astype(int)
        
    return merged

# ==========================================
# 3. PLOT: CORRELATION (GROUPED)
# ==========================================
def plot_correlation_evolution(df):
    horizons = ['1w', '1m', '2m']
    
    prefixes = ['acc_ret_', 'acc_vol_', 'dist_mean_', 'vol_', 'skew_', 'maxdd_', 'eps', 'gross_margin']
    feature_cols = [c for c in df.columns if any(p in c for p in prefixes) 
                    and 'fwd_ret' not in c 
                    and 'Target' not in c]
    
    # 1. Build DataFrame (Index=Feature, Cols=Horizons)
    evolution_data = []
    for h in horizons:
        target = f'Target_Neg_{h}'
        corr_series = df[feature_cols + [target]].corr()[target].drop(target)
        corr_series.name = h
        evolution_data.append(corr_series)
        
    corr_df = pd.concat(evolution_data, axis=1)
    
    # 2. GROUPING LOGIC (Sort by Root Name, then Window Size)
    def sort_key(feature_name):
        # Split "acc_ret_1w" into "acc_ret" and "1w"
        parts = feature_name.rsplit('_', 1)
        
        # Define window order logic
        window_rank = {'1d': 0, '1w': 1, '1m': 2, '2m': 3}
        
        if len(parts) == 2 and parts[1] in window_rank:
            root = parts[0]
            suffix = parts[1]
            rank = window_rank[suffix]
        else:
            # Handle fundamentals or other names
            root = feature_name
            rank = 99
            
        return (root, rank)

    # Sort index using custom key
    sorted_index = sorted(corr_df.index, key=sort_key)
    corr_df = corr_df.reindex(sorted_index)
    
    filename = 'feature_correlations.csv'
    # Adding index label for clarity
    corr_df.to_csv(filename, index_label='Feature')

    # 3. Plot Heatmap
    plt.figure(figsize=(10, 18)) # Taller figure to accommodate grouped list
    
    sns.heatmap(corr_df, 
                cmap='coolwarm', 
                center=0, 
                annot=True, 
                fmt=".3f", 
                linewidths=.5,
                cbar_kws={'label': 'Correlation with Crash (Negative Return)'})
    
    plt.title('Feature Signal Evolution (Grouped by Type)', fontsize=16)
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Feature Group')
    plt.tight_layout()
    
    filename = 'correlation_evolution_grouped.png'
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    prices, earnings = load_data()
    if prices is not None:
        print("Data loaded. Computing features...")
        df_full = calculate_features(prices, earnings)
        
        plot_correlation_evolution(df_full)