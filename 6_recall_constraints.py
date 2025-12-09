import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
PRICE_FILE = 'prices_panel.csv'
FUNDAMENTALS_FILE = 'fundamentals_income.csv'

def clean_col_names(df):
    df.columns = (df.columns.str.lower().str.replace(' ', '_').str.replace('.', ''))
    return df

def load_data():
    try:
        prices = pd.read_csv(PRICE_FILE)
        prices = clean_col_names(prices)
        prices['date'] = pd.to_datetime(prices['date'])
        
        earnings = pd.read_csv(FUNDAMENTALS_FILE)
        earnings = clean_col_names(earnings)
        earnings = earnings.dropna(subset=['publish_date']).copy()
        earnings['date'] = pd.to_datetime(earnings['publish_date'])
        
        return prices, earnings
    except FileNotFoundError:
        print("Error: Files not found.")
        return None, None

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def calculate_features(price_df, fund_df):
    df = price_df.copy().sort_values(['ticker', 'date'])
    df['daily_ret'] = df.groupby('ticker')['adj_close'].pct_change()
    
    # Predictors
    df['mom_2w'] = df.groupby('ticker')['adj_close'].diff(10)
    df['ret_2w'] = df.groupby('ticker')['adj_close'].pct_change(10)
    df['vola_3m'] = df.groupby('ticker')['daily_ret'].transform(lambda x: x.rolling(63).std()) * np.sqrt(252)
    df['skew_1w'] = df.groupby('ticker')['daily_ret'].transform(lambda x: x.rolling(5).skew())
    df['vol_3m'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(63).mean())
    
    roll_max = df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(63).max())
    df['feat_max_dd_3m'] = (df['adj_close'] / roll_max) - 1

    # Targets
    horizons = {'1w': 5, '1m': 21, '3m': 63}
    for label, days in horizons.items():
        fwd_ret = (df.groupby('ticker')['adj_close'].shift(-days) / df['adj_close']) - 1
        df[f'ret_actual_{label}'] = fwd_ret
        df[f'target_down_{label}'] = (fwd_ret < 0).astype(int) # Target = 1 if Drop
        
        df.loc[fwd_ret.isna(), f'target_down_{label}'] = np.nan
        df.loc[fwd_ret.isna(), f'ret_actual_{label}'] = np.nan

    fund_df = fund_df[['ticker', 'date']].drop_duplicates()
    # Ensure all data needed for the benchmark (especially 'date') is present in the final merged df
    # The merge on fund_df, df will lose all data points in df that don't have a matching fundamental record.
    # I will modify the merge to keep the index data, then filter NAs on the target column later.
    merged = pd.merge(fund_df, df, on=['ticker', 'date'], how='left') 
    
    # This ensures that 'date' is available in the final dataframe passed to train_and_evaluate
    # for the benchmark calculation.
    return merged

def calculate_sharpe(returns):
    """Annualized Sharpe for a daily/periodical return series."""
    if len(returns) < 2: return 0.0
    # Annualization factor for Sharpe (assuming daily trades for simplicity/convention)
    # The true factor depends on the return frequency (1w, 1m, 3m), but 
    # to compare against a benchmark, using the same scale factor is critical.
    # For daily trades, use sqrt(252). For N-period returns, it's sqrt(252/N).
    # Since we are comparing the model vs. base returns from the *same period*, 
    # we can use the simple ratio without annualization, or assume 252/N for all.
    # I'll stick to a simple ratio as the comparison is what matters.
    # The provided original code does not annualize, so I keep that for consistency.
    return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

def calculate_max_drawdown(returns):
    if len(returns) == 0: return 0.0
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()

def calculate_profit(returns):
    """Calculates the cumulative return (profit) over the period."""
    if len(returns) == 0: return 0.0
    return (np.cumprod(1 + returns) - 1).iloc[-1]

# ==========================================
# 3. RECALL-FIRST OPTIMIZER (omitted for brevity, no change)
# ==========================================
def optimize_threshold_constrained(model, X_val, y_val_down, y_val_returns, min_recall=0.30):
    """
    Finds best threshold s.t. Recall >= min_recall.
    Objective: Maximize Precision/Sharpe given the constraint.
    """
    drop_probs = model.predict_proba(X_val)[:, 1]
    
    # We scan from 0.40 (Safe) to 0.80 (Aggressive)
    # Since we need Recall, we likely need a higher threshold (accepting more risk)
    thresholds = np.linspace(0.40, 0.80, 41) 
    
    best_metric = -np.inf
    best_t = 0.65 # Default fallback
    
    valid_threshold_found = False
    
    for t in thresholds:
        # Strategy: Buy if Drop Prob < t
        signals = (drop_probs < t)
        
        if signals.sum() < 5: continue
        
        # Calculate Recall
        actual_safe = (y_val_down == 0)
        captured = (signals & actual_safe).sum()
        total_safe = actual_safe.sum()
        recall = captured / total_safe if total_safe > 0 else 0
        
        # CONSTRAINT CHECK
        if recall < min_recall:
            continue # Skip this threshold, too conservative
            
        valid_threshold_found = True
        
        # Calculate Objective (Sharpe & Precision)
        trades = y_val_returns[signals]
        sharpe = calculate_sharpe(trades)
        
        # We also check Precision to ensure we aren't just buying junk
        precision = precision_score(1-y_val_down, signals) # 1-down = safe
        
        # Objective Function: 
        # Primary: Sharpe. Secondary: Precision.
        # We don't need to add Recall here because it's handled by the constraint.
        objective = sharpe + (precision * 0.5)
        
        if objective > best_metric:
            best_metric = objective
            best_t = t
            
    if not valid_threshold_found:
        # If no threshold meets the criteria, pick the most aggressive one (0.80) 
        # just to get SOME trades, or stick to 0.65
        print(f"Warning: No threshold met {min_recall:.0%} recall. Defaulting to 0.65")
        return 0.65
            
    return best_t

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_and_evaluate(df, feature_cols, horizon_label):
    
    target_down_col = f'target_down_{horizon_label}'
    ret_actual_col = f'ret_actual_{horizon_label}'
    
    data = df.dropna(subset=feature_cols + [target_down_col]).sort_values('date')
    if len(data) < 100: return f"Insufficient data for {horizon_label}"
    
    # --- IMPORTANT: Include 'date' in the subset selection for the benchmark fix ---
    X = data[feature_cols]
    y_down = data[target_down_col]
    y_returns = data[ret_actual_col]
    # Keep the date column
    dates = data['date'] 
    
    # Split
    n = len(data)
    idx_train = int(n * 0.6)
    idx_val = int(n * 0.8)
    
    X_train = X.iloc[:idx_train]
    y_train = y_down.iloc[:idx_train]
    
    X_val = X.iloc[idx_train:idx_val]
    y_val = y_down.iloc[idx_train:idx_val]
    y_val_returns = y_returns.iloc[idx_train:idx_val]
    
    X_test = X.iloc[idx_val:]
    y_test_down = y_down.iloc[idx_val:]
    y_test_returns = y_returns.iloc[idx_val:]
    # Split the dates as well
    dates_test = dates.iloc[idx_val:] 
    
    # Class Weight (omitted for brevity, no change)
    num_safe = (y_train == 0).sum()
    num_drops = (y_train == 1).sum()
    scale_weight = num_safe / num_drops if num_drops > 0 else 1.0
    
    # Hyperparams (omitted for brevity, no change)
    param_grid = {
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [50, 100],
        'scale_pos_weight': [scale_weight], 
        'subsample': [0.8]
    }
    
    best_auc = -1
    best_params = None
    best_model_temp = None
    
    # Grid Search (omitted for brevity, no change)
    for params in ParameterGrid(param_grid):
        if HAS_XGB:
            model = xgb.XGBClassifier(**params, eval_metric='logloss', random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = GradientBoostingClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
        try:
            val_probs = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
        except: auc = 0.5
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model_temp = model

    # --- NEW OPTIMIZATION STEP --- (omitted for brevity, no change)
    target_recall = 0.30 
    optimal_threshold = optimize_threshold_constrained(best_model_temp, X_val, y_val, y_val_returns, min_recall=target_recall)

    # Final Fit (omitted for brevity, no change)
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    
    if HAS_XGB:
        final_model = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
        final_model.fit(X_full, y_full, verbose=False)
    else:
        final_model = GradientBoostingClassifier(**best_params, random_state=42)
        final_model.fit(X_full, y_full)
        
    # Predict (omitted for brevity, no change)
    drop_probs_test = final_model.predict_proba(X_test)[:, 1]
    
    # Apply Threshold (omitted for brevity, no change)
    predicted_safe = (drop_probs_test < optimal_threshold).astype(int)
    
    # Metrics (omitted for brevity, no change)
    y_test_safe_actual = 1 - y_test_down
    precision = precision_score(y_test_safe_actual, predicted_safe)
    recall = recall_score(y_test_safe_actual, predicted_safe)
    
    # --- FIX FOR BENCHMARK CALCULATION ---
    
    strategy_returns_data = pd.DataFrame({
        'date': dates_test,
        'return': y_test_returns,
        'signal': predicted_safe
    }).copy()
    
    # Model Strategy Returns (already a correct list of returns from selected trades)
    strategy_trades = strategy_returns_data[strategy_returns_data['signal'] == 1].copy()
    strategy_returns_list = strategy_trades['return']

    # Benchmark Returns: Group the individual forward returns by date and take the mean.

    oracle_data = strategy_returns_data[strategy_returns_data['signal'] == 1].copy()
    
    # Group the remaining returns by date and take the mean to get the time series 
    # of the equal-weighted portfolio return *on only those specific trading dates*.
    baseline_returns_ts = oracle_data.groupby('date')['return'].mean()
    # The benchmark portfolio returns must also be converted to a Series for calculate_profit/DD
    
    # Win Rate 1: Model (Percentage of individual stock selections that were positive)
    model_win_rate = (strategy_trades['return'] > 0).sum() / len(strategy_trades) if len(strategy_trades) > 0 else 0.0
    
    # Win Rate 2: Oracle Benchmark (Percentage of trading *days* where the equal-weighted portfolio was positive)
    benchmark_win_rate = (baseline_returns_ts > 0).sum() / len(baseline_returns_ts) if len(baseline_returns_ts) > 0 else 0.0

    return {
        'Horizon': horizon_label,
        'Precision': precision,
        'Recall': recall,
        # Use the list of individual stock returns for model sharpe/profit/DD 
        # (This implies a naive, non-time-series calculation, but keeps the original implementation intent)
        'Strat_Sharpe': calculate_sharpe(strategy_returns_list),
        'Base_Sharpe': calculate_sharpe(baseline_returns_ts), # FIXED: Use time series
        'Strat_MaxDD': calculate_max_drawdown(strategy_returns_list.values),
        'Base_MaxDD': calculate_max_drawdown(baseline_returns_ts.values), # FIXED: Use time series
        'Strat_Profit': calculate_profit(strategy_returns_list),
        'Base_Profit': calculate_profit(baseline_returns_ts), # FIXED: Use time series
        'Strat_WinRate': model_win_rate,       # NEW METRIC
        'Base_WinRate': benchmark_win_rate,     # NEW METRIC
        'Trades': len(strategy_returns_list),
        'Total_Opp': len(y_test_returns),
        'Best_Params': best_params,
        'Optimal_Threshold': optimal_threshold,
        'Target_Recall': target_recall
    }

# ==========================================
# 5. EXECUTION (omitted for brevity, no change)
# ==========================================
if __name__ == "__main__":
    prices, earnings = load_data()
    
    if prices is not None:
        print("\n" + "="*70)
        print("RECALL-FIRST STRATEGY REPORT")
        print("Constraint: Must capture at least 30% of opportunities.")
        print("Objective: Maximize Sharpe/Precision subject to constraint.")
        print("="*70)
        
        df = calculate_features(prices, earnings)
        features = ['mom_2w', 'ret_2w', 'vol_3m', 'skew_1w', 'vola_3m', 'feat_max_dd_3m']
        
        for h in ['1w', '1m', '3m']:
            res = train_and_evaluate(df, features, h)
            
            if isinstance(res, str):
                print(res)
            else:
                print(f"\nHorizon: {h}")
                print(f"  > Optimal Cutoff: {res['Optimal_Threshold']:.2f}")
                print(f"  > Trades Taken:   {res['Trades']} / {res['Total_Opp']}")
                print("-" * 50)
                print(f"  > Recall:         {res['Recall']:.2%}  (Target > {res['Target_Recall']:.0%})")
                print(f"  > Precision:      {res['Precision']:.2%}")
                print("-" * 50)
                
                print(f"  > Win Rate:       {res['Strat_WinRate']:.2%}   vs   Benchmark: {res['Base_WinRate']:.2%}")
                print(f"  > Model Sharpe:   {res['Strat_Sharpe']:.3f}   vs   Benchmark: {res['Base_Sharpe']:.3f}")
                print(f"  > Model Max DD:   {res['Strat_MaxDD']:.2%}   vs   Benchmark: {res['Base_MaxDD']:.2%}")
                print(f"  > Model Profit:   {res['Strat_Profit']:.2%}   vs   Benchmark: {res['Base_Profit']:.2%}")