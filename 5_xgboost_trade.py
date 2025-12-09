import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
PRICE_FILE = 'prices_panel.csv'
FUNDAMENTALS_FILE = 'fundamentals_income.csv'

def clean_col_names(df):
    """Standardize column names to snake_case."""
    df.columns = (df.columns.str.lower()
                  .str.replace(' ', '_')
                  .str.replace('.', '')
                  .str.replace(',', '')
                  .str.replace('(', '')
                  .str.replace(')', '')
                  .str.replace('/', '_')
                  .str.replace('&', ''))
    return df

def load_and_prep_data():
    print("Loading data...")
    # 1. Load Prices
    prices = pd.read_csv(PRICE_FILE)
    prices = clean_col_names(prices)
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['ticker', 'date'])
    
    # 2. Load Earnings
    earnings = pd.read_csv(FUNDAMENTALS_FILE)
    earnings = clean_col_names(earnings)
    if 'publish_date' in earnings.columns:
        earnings = earnings.rename(columns={'publish_date': 'date'})
    earnings['date'] = pd.to_datetime(earnings['date'])
    earnings = earnings.sort_values(['ticker', 'date'])
    
    # --- FUNDAMENTAL FEATURES (Calc BEFORE Merge) ---
    print("Calculating Fundamental Features...")
    # EPS Ratio (Valuation)
    share_col = 'shares_diluted' if 'shares_diluted' in earnings.columns else 'shares_basic'
    if 'net_income' in earnings.columns and share_col in earnings.columns:
        earnings['eps_raw'] = earnings['net_income'] / earnings[share_col].replace(0, np.nan)
    else:
        earnings['eps_raw'] = np.nan

    # Gross Margin
    if 'gross_profit' in earnings.columns and 'revenue' in earnings.columns:
        earnings['gross_margin'] = earnings['gross_profit'] / earnings['revenue'].replace(0, np.nan)
    else:
        earnings['gross_margin'] = np.nan

    # Drop rows that don't have a valid date
    earnings = earnings.dropna(subset=['date'])

    # --- PRICE FEATURE ENGINEERING ---
    print("Calculating Price Features (Rolling)...")
    df = prices.copy()
    
    # Windows: 1d, 1w (5d), 1m (21d), 2m (42d), 3m (63d)
    windows = {'1d': 1, '1w': 5, '1m': 21, '2m': 42, '3m': 63}
    
    for label, days in windows.items():
        # 1. Returns & Volume
        df[f'acc_ret_{label}'] = df.groupby('ticker')['adj_close'].pct_change(days)
        vol_sum = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(days).sum())
        df[f'acc_vol_{label}'] = np.log1p(vol_sum) 
        
        # 2. Statistical Features (Skip for 1d as they are noisy/undefined)
        if label != '1d':
            # Distance from Mean (Mean Reversion)
            roll_mean = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).mean())
            df[f'dist_mean_{label}'] = (df['adj_close'] / roll_mean) - 1
            
            # Max Drawdown (Risk)
            roll_min = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).min())
            roll_max = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(days).max())
            df[f'maxdd_{label}'] = (roll_min / roll_max) - 1

        # 3. Volatility & Skew (Needs >1 day)
        if days > 1:
            df[f'vol_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).std()) * np.sqrt(252)
            df[f'skew_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).skew())

    # --- TARGETS (Forward Returns) ---
    print("Calculating Targets...")
    horizons = {'1w': 5, '1m': 21, '3m': 63}
    for label, days in horizons.items():
        # Return from Today to Today+Horizon
        df[f'fwd_ret_{label}'] = df.groupby('ticker')['adj_close'].shift(-days) / df['adj_close'] - 1

    # --- MERGE LOGIC (Align T+1) ---
    print("Aligning Trades to T+1 Post-Earnings...")
    
    # 1. Set Entry Date (Filter out T+1 reaction)
    earnings['trade_date_target'] = earnings['date'] + pd.Timedelta(days=1)
    
    # 2. Sort keys for merge_asof
    earnings = earnings.sort_values('trade_date_target')
    df = df.sort_values('date')
    
    # 3. Merge AsOf (Forward look to find valid trading day)
    merged = pd.merge_asof(earnings, df, 
                           left_on='trade_date_target', 
                           right_on='date', 
                           by='ticker', 
                           direction='forward', # Finds D+3, or next trading day
                           suffixes=('_fund', '_price'))
    
    merged = merged.dropna(subset=['adj_close'])
    
    # --- VALUATION RATIOS (Point-in-Time) ---
    # Now we have EPS (Fund) and Price (Entry Date)
    merged['earnings_yield'] = merged['eps_raw'] / merged['adj_close']
    merged['pe_ratio'] = merged['adj_close'] / merged['eps_raw'].replace(0, np.nan)
    
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- BINARY TARGETS ---
    for h in horizons.keys():
        # Neg = Drop < -2% (Safety Target)
        merged[f'target_neg_{h}'] = (merged[f'fwd_ret_{h}'] < -0.02).astype(int)
        # Outcome = Positive Return (For Win Rate Calc)
        merged[f'outcome_pos_{h}'] = (merged[f'fwd_ret_{h}'] > 0).astype(int)
        
    print(f"Data Prep Complete. Shape: {merged.shape}")
    return merged

# ==========================================
# 2. MODEL TRAINING
# ==========================================
def train_event_model(df):
    
    # --- FEATURE SET DEFINITIONS ---
    # Mapping user request to actual column names
    feature_sets = {
        '1w': [
            'earnings_yield',   # eps/valuation
            'vol_1w',           # vol-2m
            'skew_1w',          # skew 1w
            'maxdd_3m',         # maxdd3m
            'dist_mean_1w',     # dist-mean-1w
            'acc_ret_1d',       # return 1d
            'acc_ret_1m'        # return 1m
        ],
        '1m': [
            'earnings_yield',   # eps/valuation
            'vol_1m',           # vol-2m
            'skew_1w',          # skew 1w
            'gross_margin',     # gross margin
            'dist_mean_1w',     # dist-mean-1w
            'acc_ret_1m'        # return 1m
        ],
        '3m': [
            'earnings_yield',   # eps/valuation
            'vol_3m',           # vol-1w
            'skew_1w',          # skew 1w
            'maxdd_3m',         # maxdd3m
            'dist_mean_3m',     # dist-mean-3m
            'acc_ret_3m'        # return 3m
        ]
    }

    horizons = ['1w', '1m', '3m']

    for horizon in horizons:
        target = f'target_neg_{horizon}'
        outcome_col = f'outcome_pos_{horizon}'
        features = feature_sets[horizon]
        
        # Verify features exist
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            print(f"Skipping {horizon}: Missing columns {missing_feats}")
            continue

        print(f"\n{'='*60}")
        print(f"TRAINING HORIZON: {horizon}")
        print(f"Features: {features}")
        print(f"{'='*60}")
        
        # Drop NAs
        data = df.dropna(subset=features + [target, outcome_col]).sort_values('date_price')
        
        # Chronological Split
        n = len(data)
        idx_train = int(n * 0.6)
        idx_val = int(n * 0.8)
        
        X_train = data.iloc[:idx_train][features]
        y_train = data.iloc[:idx_train][target]
        
        X_val = data.iloc[idx_train:idx_val][features]
        y_val = data.iloc[idx_train:idx_val][target]
        
        X_test = data.iloc[idx_val:][features]
        y_test = data.iloc[idx_val:][target]             # Target: Drop?
        y_test_outcome = data.iloc[idx_val:][outcome_col] # Actual: Up?
        y_test_ret = data.iloc[idx_val:][f'fwd_ret_{horizon}']

        # Train XGBoost
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,            
            scale_pos_weight=scale,
            eval_metric='logloss',
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"Best Validation LogLoss: {model.best_score if hasattr(model, 'best_score') else 'N/A'}")

        # Evaluate
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # --- BENCHMARK METRICS ---
        total_possible_trades = len(X_test)
        market_win_rate = y_test_outcome.mean()
        
        print(f"\n--- Results for {horizon} Horizon ---")
        print(f"Market Win Rate (Baseline): {market_win_rate:.2%}")
        print(f"Total Possible Trades:      {total_possible_trades}")
        print("-" * 85)
        print(f"{'Threshold':<10} | {'Trades':<8} | {'% Coverage':<12} | {'Win Rate':<10} | {'Edge':<8} | {'Avg Return':<10}")
        print("-" * 85)
        
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        
        for t in thresholds:
            # Signal: Buy if Probability of DROP is LOW (< t)
            buy_mask = test_probs < t
            
            if buy_mask.sum() > 0:
                count = buy_mask.sum()
                coverage = count / total_possible_trades
                
                win_rate = y_test_outcome[buy_mask].mean()
                edge = win_rate - market_win_rate
                avg_ret = y_test_ret[buy_mask].mean()
                
                print(f"{t:<10.2f} | {count:<8} | {coverage:<12.1%} | {win_rate:<10.2%} | {edge:<+8.2%} | {avg_ret:<+10.2%}")
            else:
                print(f"{t:<10.2f} | 0        | 0.0%         | N/A        | N/A      | N/A")

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    df_final = load_and_prep_data()
    
    if df_final is not None:
        train_event_model(df_final)