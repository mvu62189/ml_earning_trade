import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import itertools
from sklearn.metrics import log_loss

# ==========================================
# 1. DATA LOADING (Standardized)
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

def load_and_merge_data():
    """
    Loads daily prices and merges the MOST RECENT fundamentals 
    to every single trading day (Forward Fill).
    """
    
    # 1. Load Prices
    prices = pd.read_csv(PRICE_FILE)
    prices = clean_col_names(prices)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['date', 'ticker']) # for merge_asof
    
    # 2. Load Fundamentals
    earnings = pd.read_csv(FUNDAMENTALS_FILE)
    earnings = clean_col_names(earnings)
    earnings = earnings.loc[:, ~earnings.columns.duplicated()]
    
    if 'publish_date' in earnings.columns:
        earnings = earnings.rename(columns={'publish_date': 'date'})
    
    earnings['date'] = pd.to_datetime(earnings['date'])
    earnings = earnings.dropna(subset=['date']).sort_values(['date', 'ticker'])
    
    # Calculate Fundamentals (EPS & Margin)
    share_col = 'shares_diluted' if 'shares_diluted' in earnings.columns else 'shares_basic'
    
    # Division for EPS
    if 'net_income' in earnings.columns and share_col in earnings.columns:
        earnings['eps'] = earnings['net_income'] / earnings[share_col].replace(0, np.nan)
    else:
        earnings['eps'] = np.nan

    # Division for Margin
    if 'gross_profit' in earnings.columns and 'revenue' in earnings.columns:
        earnings['gross_margin'] = earnings['gross_profit'] / earnings['revenue'].replace(0, np.nan)
    else:
        earnings['gross_margin'] = np.nan

    fund_subset = earnings[['date', 'ticker', 'eps', 'gross_margin']].dropna()

    # 3. MERGE ASOF
    print("Merging fundamentals to daily data (Forward Fill)...")
    merged = pd.merge_asof(prices, fund_subset, 
                            on='date', 
                            by='ticker', 
                            direction='backward') # Look backwards for the last report
    return merged
        

# ==========================================
# 2. FEATURE ENGINEERING (Continuous)
# ==========================================
def calculate_features(df):
    """
    Calculates features on the continuous daily timeframe.
    """
    # Sort for rolling calcs
    df = df.sort_values(['ticker', 'date'])
    
    print("Calculating rolling features on full history...")
    
    # 1. Danger Signals (Positive Correlation with Negative returns)
    # Momentum (1 week)
    df['mom_1w'] = df.groupby('ticker')['adj_close'].diff(5)
    # Max Drawdown (1 month)
    roll_min = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(21).min())
    roll_max = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(21).max())
    df['maxdd_1m'] = (roll_min / roll_max) - 1
    # Skew (3 months)
    df['skew_3m'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(63).skew())
    # Price Position in Range (Stochastic Proxy)
    df['price_pct_1m'] = (df['adj_close'] - roll_min) / (roll_max - roll_min)
    # 2. Protective Signals (Negative Correlation with Crash)
    # Volume Trend
    df['volume_1w'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
    # Volatility (1 month)
    df['vol_1m'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(21).std()) * np.sqrt(252)

    # 3. Targets (3m / 63 days Horizon)
    # horizon = 63
    
    horizons = {
        '1d': 1,
        '1w': 5,
        '1m': 21,
        '3m': 63
    }
    for h_name, h_days in horizons.items():
        # Calculate Forward Return
        col_ret = f'fwd_ret_{h_name}'
        df[col_ret] = df.groupby('ticker')['adj_close'].shift(-h_days) / df['adj_close'] - 1
        
        # Calculate Binary Target (Drop vs No Drop)
        col_target = f'target_neg_{h_name}'
        # 1 if return is negative, 0 otherwise
        df[col_target] = (df[col_ret] < 0).astype(int)
        
        # Calculate Binary Outcome (Positive Return - for win rate calc)
        col_outcome = f'target_pos_{h_name}'
        df[col_outcome] = (df[col_ret] > 0).astype(int)
        
    #df['fwd_ret_3m'] = df.groupby('ticker')['adj_close'].shift(-horizon) / df['adj_close'] - 1
    
    #df = df.dropna(subset=['fwd_ret_3m'])
    
    # Binary Targets
    #df['target_neg'] = (df['fwd_ret_3m'] < 0).astype(int)
    #df['target_pos'] = (df['fwd_ret_3m'] > 0).astype(int)
    
    return df

# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================

def train_continuous_model(df):
    features = ['mom_1w', 'maxdd_1m', 'skew_3m', 'price_pct_1m', 
                'volume_1w', 'vol_1m', 'eps', 'gross_margin']
    
    # Define the horizons we created earlier
    horizon_map = {
        '1d': {'days': 1,  'subsample': 1},  # No subsampling needed for 1d
        '1w': {'days': 5,  'subsample': 2},  # Light subsampling
        '1m': {'days': 21, 'subsample': 5},  # Medium subsampling
        '3m': {'days': 63, 'subsample': 10}  # Heavy subsampling (original)
    }

    # Grid for tuning
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 5],
        'gamma': [0.1, 0.2, 0.5]
    }
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # --- MASTER LOOP OVER HORIZONS ---
    for h_name, config in horizon_map.items():
        target_col = f'target_neg_{h_name}'
        outcome_col = f'target_pos_{h_name}'
        
        print(f"\n{'='*40}")
        print(f" TRAINING MODEL FOR HORIZON: {h_name.upper()}")
        print(f"{'='*40}")
        
        # Filter Data for this specific target
        # We must drop rows where THIS horizon's target is NaN
        data = df.dropna(subset=features + [target_col]).sort_values('date')
        
        # Chronological Split
        n = len(data)
        idx_train = int(n * 0.6)
        idx_val = int(n * 0.8)
        
        # 1. Training Set (Dynamic Subsampling)
        # We use the subsample rate defined in the map above
        step = config['subsample']
        train_subset = data.iloc[:idx_train].iloc[::step]
        
        X_train = train_subset[features]
        y_train = train_subset[target_col]
        
        # 2. Validation & Test
        X_val = data.iloc[idx_train:idx_val][features]
        y_val = data.iloc[idx_train:idx_val][target_col]
        
        X_test = data.iloc[idx_val:][features]
        y_test = data.iloc[idx_val:][target_col]
        y_test_outcome = data.iloc[idx_val:][outcome_col]
        
        # --- GRID SEARCH (Mini Version) ---
        best_score = float('inf')
        best_model = None
        scale = (y_train == 0).sum() / (y_train == 1).sum()

        for params in param_combinations:
            model = xgb.XGBClassifier(
                n_estimators=1000,
                scale_pos_weight=scale,
                eval_metric='logloss',
                early_stopping_rounds=20,
                n_jobs=-1,
                random_state=42,
                **params
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Score
            val_probs = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, val_probs)
            
            if score < best_score:
                best_score = score
                best_model = model

        print(f"Best Validation LogLoss: {best_score:.4f}")
        
        # --- FINAL EVALUATION ---
        test_probs = best_model.predict_proba(X_test)[:, 1]
        baseline_win_rate = y_test_outcome.mean()
        
        print(f"\nResults for {h_name} Horizon (Baseline Win Rate: {baseline_win_rate:.2%})")
        print(f"{'Safety Thresh':<15} | {'Trades/Day':<12} | {'Win Rate':<10} | {'Edge':<10}")
        print("-" * 60)
        
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        for t in thresholds:
            buy_signal = (test_probs < t)
            total_trades = buy_signal.sum()
            # Calculate days based on specific test set duration
            days_in_test = (data.iloc[idx_val:]['date'].max() - data.iloc[idx_val:]['date'].min()).days
            trades_per_day = total_trades / days_in_test if days_in_test > 0 else 0
            
            if total_trades > 0:
                win_rate = y_test_outcome[buy_signal].mean()
                edge = win_rate - baseline_win_rate
            else:
                win_rate = 0.0
                edge = 0.0
                
            print(f"{t:<15.2f} | {trades_per_day:<12.1f} | {win_rate:<10.2%} | {edge:+.2%}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    df_merged = load_and_merge_data()
    
    if df_merged is not None:
        df_features = calculate_features(df_merged)
        train_continuous_model(df_features)