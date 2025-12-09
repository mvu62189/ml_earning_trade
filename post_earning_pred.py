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
    # EPS
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
    
    # Windows: 1d (Momentum), 1w (Weekly), 1m (Monthly), 2m (Quarterly-ish)
    windows = {'1d': 1, '1w': 5, '1m': 21, '2m': 42}
    
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

        # 3. Volatility & Skew
        if days > 1:
            df[f'vol_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).std()) * np.sqrt(252)
            df[f'skew_{label}'] = df.groupby('ticker')['adj_close'].pct_change().transform(lambda x: x.rolling(days).skew())

    # --- TARGETS (Forward Returns) ---
    print("Calculating Targets...")
    horizons = {'1w':5, '1m': 21, '3m': 63}
    for label, days in horizons.items():
        # Return from Today to Today+Horizon
        df[f'fwd_ret_{label}'] = df.groupby('ticker')['adj_close'].shift(-days) / df['adj_close'] - 1

    # --- MERGE LOGIC (Align T+1) ---
    print("Aligning Trades to T+1 Post-Earnings...")
    
    # 1. Set Entry Date
    earnings['trade_date_target'] = earnings['date'] + pd.Timedelta(days=1)
    
    # 2. Sort keys for merge_asof
    earnings = earnings.sort_values('trade_date_target')
    df = df.sort_values('date')
    
    # 3. Merge AsOf (Forward look to find valid trading day)
    # This pulls the price features (acc_ret, vol) AND the targets (fwd_ret) 
    # from the specific entry day row.
    merged = pd.merge_asof(earnings, df, 
                           left_on='trade_date_target', 
                           right_on='date', 
                           by='ticker', 
                           direction='forward', # Finds D+3, or D+4/5 if weekend
                           suffixes=('_fund', '_price'))
    
    # Drop rows where we couldn't find a price (e.g. delisted right after earnings)
    merged = merged.dropna(subset=['adj_close'])
    
    # --- VALUATION RATIOS (Point-in-Time) ---
    # Now we have EPS (Fund) and Price (Entry Date)
    merged['earnings_yield'] = merged['eps_raw'] / merged['adj_close']
    merged['pe_ratio'] = merged['adj_close'] / merged['eps_raw'].replace(0, np.nan)
    
    # Clean Infinite values
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- BINARY TARGETS ---
    for h in horizons.keys():
        # Neg = Drop < -2%
        merged[f'target_neg_{h}'] = (merged[f'fwd_ret_{h}'] < -0.02).astype(int)
        
    print(f"Data Prep Complete. Shape: {merged.shape}")
    return merged

# ==========================================
# 2. MODEL TRAINING
# ==========================================
def train_event_model(df, horizon='3m'):
    # Define Target
    target = f'target_neg_{horizon}'
    
    # Define Feature Candidates
    # We list everything we *think* we calculated.
    feature_candidates = [
        # Fundamentals
        'earnings_yield', 'pe_ratio',
        # Price (1w)
        'dist_mean_1w', 'maxdd_1w', 'vol_1w', 'skew_1w',
        # Price (1m)
        'dist_mean_1m', 'vol_1m', 'skew_1m',
        # Price (2m)
        'acc_ret_2m', 'dist_mean_2m', 'maxdd_2m', 'vol_2m', 'skew_2m'
    ]
    
    # Robust Filter: Only keep columns that actually exist
    features = [f for f in feature_candidates if f in df.columns]
    
    print(f"\nTraining for Horizon: {horizon}")
    # print(f"Features Selected ({len(features)}): {features}")
    
    # Drop NAs for training
    data = df.dropna(subset=features + [target]).sort_values('date_price')
    
    print(f"Training Samples: {len(data)}")
    
    # Chronological Split
    n = len(data)
    idx_train = int(n * 0.6)
    idx_val = int(n * 0.8)
    
    X_train = data.iloc[:idx_train][features]
    y_train = data.iloc[:idx_train][target]
    
    X_val = data.iloc[idx_train:idx_val][features]
    y_val = data.iloc[idx_train:idx_val][target]
    
    X_test = data.iloc[idx_val:][features]
    y_test = data.iloc[idx_val:][target]
    y_test_ret = data.iloc[idx_val:][f'fwd_ret_{horizon}']

    # Train XGBoost
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        scale_pos_weight=scale,
        eval_metric='logloss',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    print(f"Best Validation LogLoss: {model.best_score if hasattr(model, 'best_score') else 'N/A'}")

    # Evaluate
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Strategy Check
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    print(f"\n{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Avg Return':<10}")
    print("-" * 50)
    
    for t in thresholds:
        buy_mask = test_probs < t
        if buy_mask.sum() > 0:
            count = buy_mask.sum()
            win_rate = (y_test_ret[buy_mask] > 0).mean()
            avg_ret = y_test_ret[buy_mask].mean()
            print(f"{t:<10.2f} | {count:<8} | {win_rate:<10.2%} | {avg_ret:+.2%}")
        else:
            print(f"{t:<10.2f} | 0        | N/A        | N/A")

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    df_final = load_and_prep_data()
    
    if df_final is not None:
        train_event_model(df_final, horizon='3m')
        train_event_model(df_final, horizon='1m')
        train_event_model(df_final, horizon='1w')