import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid

# File Names
DF_FILE = 'feature_panel_daily.csv'
TARGET_HORIZON = '3m' # Default to 3m horizon

def calculate_sharpe(returns):
    """Calculates Sharpe Ratio."""
    if len(returns) < 2: return 0.0
    return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

# ==========================================
# 1. DATA LOADING (Uses prepared daily file)
# ==========================================
def load_data():
    """Loads the prepared daily feature panel."""
    try:
        df = pd.read_csv(DF_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        
        return df
    except FileNotFoundError:
        print(f"Error: {DF_FILE} not found.")
        return None

# ==========================================
# 2. DYNAMIC RANK OPTIMIZER
# ==========================================
def optimize_rank_threshold(model, X_val, df_val, fwd_ret_col):
    """
    Finds the optimal Daily Percentile Rank (P%) by maximizing Sharpe Ratio 
    on the validation set.
    """
    probs = model.predict_proba(X_val)[:, 1] # Prob(Down)
    temp = df_val.copy()
    temp['prob_down'] = probs
    
    # Rank: 0.0 = Safest (Lowest Prob)
    # Ranks the stocks cross-sectionally (per day)
    temp['safety_rank'] = temp.groupby('date')['prob_down'].rank(pct=True)
    
    percentiles = [0.10, 0.20, 0.30, 0.40, 0.50]
    best_sharpe = -np.inf
    best_pct = 0.20 # Default fallback
    
    for p in percentiles:
        # Strategy: Buy if in top p% of safety
        signals = temp['safety_rank'] <= p
        
        if signals.sum() < 50: continue
        
        returns = temp.loc[signals, fwd_ret_col]
        sharpe = calculate_sharpe(returns)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_pct = p
            
    return best_pct

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_dynamic_strategy(df):
    
    # Comprehensive Feature List (Corrected and robust names)
    features = [
        'earnings_yield', 'gross_margin', 'pe_ratio',
        'acc_ret_1w', 'vola_1m', 'skew_1m', 'max_dd_3m', 'vol_1m'
    ]
    target = f'target_neg_{TARGET_HORIZON}'
    fwd_ret = f'fwd_ret_{TARGET_HORIZON}'
    
    # Filter only samples where all features and target are present
    data = df.dropna(subset=features + [target]).sort_values('date')
    
    # Split
    n = len(data)
    idx_train = int(n * 0.6)
    idx_val = int(n * 0.8)
    
    train = data.iloc[:idx_train]
    val = data.iloc[idx_train:idx_val]
    test = data.iloc[idx_val:]
    
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test = test[features] 
    
    # Model Setup
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos == 0 or n_neg == 0:
        print("SKIP: Training set has only one class.")
        return

    scale = n_neg / n_pos
    
    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        scale_pos_weight=scale, eval_metric='logloss', 
        early_stopping_rounds=20, random_state=42, n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # --- OPTIMIZATION (Finds best P%) ---
    best_pct = optimize_rank_threshold(model, X_val, val, fwd_ret)
    
    # --- FINAL TEST ---
    test_probs = model.predict_proba(X_test)[:, 1]
    test_df = test.copy()
    test_df['prob_down'] = test_probs
    
    # Dynamic Rank Signal
    test_df['safety_rank'] = test_df.groupby('date')['prob_down'].rank(pct=True)
    test_df['signal'] = test_df['safety_rank'] <= best_pct
    
    # Results
    trades = test_df[test_df['signal']]
    
    if len(trades) == 0:
        print("No trades triggered in test set.")
        return
    
    strategy_win_rate = (trades[fwd_ret] > 0).mean()
    strategy_avg_ret = trades[fwd_ret].mean()
    strategy_sharpe = calculate_sharpe(trades[fwd_ret])
    
    # --- 2. BENCHMARK METRICS (Long Only All Stocks) ---
    benchmark_returns = test_df[fwd_ret]
    benchmark_win_rate = (benchmark_returns > 0).mean()
    benchmark_avg_ret = benchmark_returns.mean()
    benchmark_sharpe = calculate_sharpe(benchmark_returns)
    
    # --- OUTPUT COMPARISON (CORRECTED) ---
    print("\n" + "="*70)
    print(f"OUT-OF-SAMPLE RESULTS | Horizon: {TARGET_HORIZON} (Daily Entry)")
    print("="*70)
    print(f"Optimal Strategy: Buy Top {best_pct:.0%} Safest Stocks Daily")
    print("-" * 70)
    
    print(f"{'Metric':<20} | {'Strategy':<15} | {'Benchmark (All)'}")
    print("-" * 70)
    
    # Corrected Line 1: Total Trades (Use formatting on the right side)
    print(f"{'Total Trades':<20} | {len(trades):<15} | {len(test):<15} (100% Coverage)")
    
    # Corrected Line 2: Win Rate
    # The width 15 must contain the result including the '%' sign.
    print(f"{'Win Rate':<20} | {strategy_win_rate:<15.2%} | {benchmark_win_rate:.2%}") 
    
    # Corrected Line 3: Avg Return
    print(f"{'Avg Return':<20} | {strategy_avg_ret:<15.2%} | {benchmark_avg_ret:.2%}")
    
    # Corrected Line 4: Sharpe Ratio
    print(f"{'Sharpe Ratio':<20} | {strategy_sharpe:<15.3f} | {benchmark_sharpe:.3f}")

    print("\nSTRATEGY EDGE OVER BENCHMARK:")
    print(f"  > Win Rate Edge: {strategy_win_rate - benchmark_win_rate:+.2%}")
    print(f"  > Sharpe Edge:   {strategy_sharpe - benchmark_sharpe:+.3f}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    df_merged = load_data()
    if df_merged is not None:
        horizons = {'1w': 5, '1m': 21, '3m': 63}
        for h in horizons.keys():
            # Target is 1 if forward return is less than -2%
            df_merged[f'target_neg_{h}'] = (df_merged[f'fwd_ret_{h}'] < -0.02).astype(int)
        train_dynamic_strategy(df_merged)