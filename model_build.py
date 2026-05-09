# =====  DIVERSE SAMPLING + BAYESIAN HYPERPARAMETER OPTIMIZATION (XGBOOST ONLY) - MULTI STOCK =====
import pandas as pd, numpy as np, warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os
import glob

# Install required packages:
# pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Integer, Real

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# 1.  CONFIGURATION
DATA_FOLDER = "Nifty_Data"  # Folder containing all stock CSV files
MODELS_FOLDER = "Nifty_models"  # Folder to save models
RESULTS_FILE = "Nifty_models/stock_results.txt"  # Results summary file

DATE_COL = "date"
TARGET_COL = "target"
FEATURES = ['open','high','low','close','volume',
            'ATR_21','EMA_9','EMA_21','SMA_21','SMA_42',
            'RSI_14','BB_upper','BB_lower','OBV','CMF_21','VWAP']

BUY_RATIO = 0.5        # 50% Buy signals
NOT_BUY_RATIO = 0.5     # 50% Not-Buy signals
N_CLUSTERS = 50         # Number of clusters for diverse sampling
RANDOM_STATE = 42
N_ITER = 10             # Bayesian optimization iterations
LOOKBACK_PERIODS = 5    # Number of lookback periods to add

# ---------------------------------------------------------------
# 2.  DATA CLEANING FUNCTION
def clean_data_remove_nan(df, features, target_col):
    """Remove rows with ANY NaN values in features or target column"""
    print("=== DATA CLEANING - REMOVING NaN VALUES ===")

    original_size = len(df)
    print(f"Original dataset size: {original_size:,} rows")

    # Check for NaN in each column
    print(f"\nNaN counts per column:")
    for col in features + [target_col]:
        if col in df.columns:
            nan_count = df[col].isnull().sum()
            print(f"  {col}: {nan_count:,} NaN values ({nan_count/len(df)*100:.2f}%)")

    # Remove rows with ANY NaN in features or target
    columns_to_check = [col for col in features + [target_col] if col in df.columns]
    clean_df = df.dropna(subset=columns_to_check).copy()

    removed_rows = original_size - len(clean_df)
    print(f"\nRows removed due to NaN: {removed_rows:,} ({removed_rows/original_size*100:.2f}%)")
    print(f"Clean dataset size: {len(clean_df):,} rows")

    # Verify no NaN values remain
    remaining_nan = clean_df[columns_to_check].isnull().sum().sum()
    if remaining_nan == 0:
        print("✅ All NaN values successfully removed")
    else:
        print(f"⚠️  Warning: {remaining_nan} NaN values still remain")

    return clean_df

# ---------------------------------------------------------------
# 3.  DIVERSE SAMPLING FUNCTION (MODIFIED FOR FULL DATASET)
def diverse_temporal_sampling_full_dataset(df, target_col, buy_class, not_buy_class,
                                         features, buy_ratio=0.4, n_clusters=50, temporal_col='date'):
    """MODIFIED: Use entire dataset - ALL Buy samples + diverse Not-Buy samples for 60:40 ratio"""
    print("=== DIVERSE SAMPLING STRATEGY - FULL DATASET (50% Buy : 50% Not-Buy) ===")

    buy_df = df[df[target_col] == buy_class].copy()
    not_buy_df = df[df[target_col] == not_buy_class].copy()
    n_buy = len(buy_df)
    n_not_buy_target = int(n_buy * (1-buy_ratio) / buy_ratio)
    n_not_buy_target = min(n_not_buy_target, len(not_buy_df))

    print(f"Full dataset analysis:")
    print(f"  Buy signals: {n_buy:,}")
    print(f"  Not-Buy signals: {len(not_buy_df):,}")
    print(f"\nTarget balanced dataset:")
    print(f"  Buy signals: {n_buy:,} (50%)")
    print(f"  Not-Buy signals to select: {n_not_buy_target:,} (50%)")
    print(f"  Total final size: {n_buy + n_not_buy_target:,}")

    # Temporal diversification
    not_buy_df[temporal_col] = pd.to_datetime(not_buy_df[temporal_col])
    not_buy_df['year'] = not_buy_df[temporal_col].dt.year
    not_buy_df['month'] = not_buy_df[temporal_col].dt.month
    not_buy_df['year_month'] = not_buy_df['year'].astype(str) + '_' + not_buy_df['month'].astype(str).str.zfill(2)

    # Clustering for feature diversity (NO NaN filling - data is already clean)
    not_buy_features = not_buy_df[features]
    effective_clusters = min(n_clusters, n_not_buy_target // 5, len(not_buy_df) // 10)
    effective_clusters = max(1, effective_clusters)

    print(f"\nClustering Not-Buy samples:")
    print(f"  Using {effective_clusters} clusters for diversity")
    print(f"  Feature matrix shape: {not_buy_features.shape}")

    kmeans = KMeans(n_clusters=effective_clusters, random_state=RANDOM_STATE, n_init=10)
    not_buy_df['cluster_id'] = kmeans.fit_predict(not_buy_features)

    # Combined sampling strategy
    sampled_not_buy = pd.DataFrame()
    temporal_groups = not_buy_df.groupby('year_month')
    samples_per_period = max(1, n_not_buy_target // len(temporal_groups))

    print(f"Temporal sampling strategy:")
    print(f"  Time periods found: {len(temporal_groups)}")
    print(f"  Target samples per period: {samples_per_period}")

    for period, period_df in temporal_groups:
        if len(period_df) == 0:
            continue
        period_clusters = period_df['cluster_id'].unique()
        samples_per_cluster_period = max(1, samples_per_period // len(period_clusters))

        for cluster_id in period_clusters:
            cluster_period_df = period_df[period_df['cluster_id'] == cluster_id]
            n_to_sample = min(len(cluster_period_df), samples_per_cluster_period)
            if n_to_sample > 0:
                sampled = cluster_period_df.sample(n=n_to_sample, random_state=RANDOM_STATE)
                sampled_not_buy = pd.concat([sampled_not_buy, sampled])

    # Fill remainder if needed
    if len(sampled_not_buy) < n_not_buy_target:
        remaining_needed = n_not_buy_target - len(sampled_not_buy)
        remaining_pool = not_buy_df[~not_buy_df.index.isin(sampled_not_buy.index)]
        if len(remaining_pool) > 0:
            additional = remaining_pool.sample(n=min(remaining_needed, len(remaining_pool)),
                                             random_state=RANDOM_STATE)
            sampled_not_buy = pd.concat([sampled_not_buy, additional])

    # Trim if too many
    if len(sampled_not_buy) > n_not_buy_target:
        sampled_not_buy = sampled_not_buy.sample(n=n_not_buy_target, random_state=RANDOM_STATE)

    # Combine and shuffle
    balanced_df = pd.concat([buy_df, sampled_not_buy])
    balanced_df = balanced_df.drop(columns=['cluster_id', 'year', 'month', 'year_month'], errors='ignore')
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Verify final distribution
    final_counts = balanced_df[target_col].value_counts()
    final_buy_ratio = final_counts[buy_class] / len(balanced_df)
    final_not_buy_ratio = final_counts[not_buy_class] / len(balanced_df)

    print(f"\n=== FINAL BALANCED DATASET ===")
    print(f"Total rows: {len(balanced_df):,}")
    print(f"Buy signals: {final_counts[buy_class]:,} ({final_buy_ratio*100:.1f}%)")
    print(f"Not-Buy signals: {final_counts[not_buy_class]:,} ({final_not_buy_ratio*100:.1f}%)")

    return balanced_df

# ---------------------------------------------------------------
# 4.  LOOKBACK FEATURES GENERATION FUNCTION
def add_lookback_features(df, features, lookback_periods=5, date_col='date'):
    """Add lookback features for previous N periods"""
    print("\n=== ADDING LOOKBACK FEATURES ===")
    print(f"Creating {lookback_periods} lookback periods for {len(features)} features")
    
    # Sort by date to ensure proper order
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    
    # Create lookback features
    lookback_features = []
    for period in range(1, lookback_periods + 1):
        for feature in features:
            lookback_col = f"{feature}_lag_{period}"
            df_sorted[lookback_col] = df_sorted[feature].shift(period)
            lookback_features.append(lookback_col)
    
    print(f"Added {len(lookback_features)} lookback features:")
    print(f"  Features per period: {len(features)}")
    print(f"  Total lookback features: {len(lookback_features)}")
    
    # Remove rows with NaN values caused by shifting (first few rows)
    original_size = len(df_sorted)
    df_with_lookback = df_sorted.dropna().copy()
    removed_rows = original_size - len(df_with_lookback)
    
    print(f"\nRows removed due to lookback NaN: {removed_rows:,}")
    print(f"Final dataset size: {len(df_with_lookback):,}")
    
    # Return updated feature list and dataframe
    all_features = features + lookback_features
    
    print(f"\nTotal features for modeling: {len(all_features)}")
    print(f"  Original features: {len(features)}")
    print(f"  Lookback features: {len(lookback_features)}")
    
    return df_with_lookback, all_features

# ---------------------------------------------------------------
# 5.  ENHANCED EVALUATION FUNCTION
def evaluate_model_comprehensive(model, X_test, y_test, model_name, rank, buy_class, not_buy_class):
    """Comprehensive model evaluation with all metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate all metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=buy_class)
    recall = recall_score(y_test, y_pred, pos_label=buy_class)
    f1 = f1_score(y_test, y_pred, pos_label=buy_class)
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Not-Buy', 'Buy'], output_dict=True)

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - RANK {rank} RESULTS")
    print(f"{'='*60}")

    print(f"📊 PERFORMANCE METRICS:")
    print(f"   AUC Score: {auc:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    print(f"\n📈 CONFUSION MATRIX:")
    print("                    Predicted")
    print("                    Not-Buy  Buy")
    print(f"Actual Not-Buy        {cm[not_buy_class,not_buy_class]:4d}   {cm[not_buy_class,buy_class]:4d}")
    print(f"Actual Buy            {cm[buy_class,not_buy_class]:4d}   {cm[buy_class,buy_class]:4d}")

    # Additional insights
    tn, fp, fn, tp = cm[not_buy_class,not_buy_class], cm[not_buy_class,buy_class], cm[buy_class,not_buy_class], cm[buy_class,buy_class]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n🎯 DETAILED ANALYSIS:")
    print(f"   True Positives (Buy correctly predicted):  {tp}")
    print(f"   True Negatives (Not-Buy correctly predicted): {tn}")
    print(f"   False Positives (Wrong Buy predictions): {fp}")
    print(f"   False Negatives (Missed Buy opportunities): {fn}")
    print(f"   Specificity (True Negative Rate): {specificity:.4f}")

    print(f"\n📋 CLASSIFICATION REPORT:")
    # Handle classification report keys safely
    try:
        # Try different possible key formats
        not_buy_key = str(not_buy_class) if str(not_buy_class) in class_report else not_buy_class
        buy_key = str(buy_class) if str(buy_class) in class_report else buy_class

        if not_buy_key in class_report and buy_key in class_report:
            print(f"   Not-Buy Class - Precision: {class_report[not_buy_key]['precision']:.3f}, Recall: {class_report[not_buy_key]['recall']:.3f}, F1: {class_report[not_buy_key]['f1-score']:.3f}")
            print(f"   Buy Class     - Precision: {class_report[buy_key]['precision']:.3f}, Recall: {class_report[buy_key]['recall']:.3f}, F1: {class_report[buy_key]['f1-score']:.3f}")
        else:
            # Fallback: show available keys and use them
            available_keys = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            print(f"   Available class keys: {available_keys}")
            if len(available_keys) >= 2:
                key1, key2 = available_keys[0], available_keys[1]
                print(f"   Class {key1} - Precision: {class_report[key1]['precision']:.3f}, Recall: {class_report[key1]['recall']:.3f}, F1: {class_report[key1]['f1-score']:.3f}")
                print(f"   Class {key2} - Precision: {class_report[key2]['precision']:.3f}, Recall: {class_report[key2]['recall']:.3f}, F1: {class_report[key2]['f1-score']:.3f}")
    except Exception as e:
        print(f"   Error displaying classification report: {e}")
        print(f"   Available keys: {list(class_report.keys())}")

    # Calculate Not-Buy precision for results summary
    not_buy_precision = precision_score(y_test, y_pred, pos_label=not_buy_class)

    return {
        'auc': auc, 'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'confusion_matrix': cm,
        'classification_report': class_report, 'specificity': specificity,
        'not_buy_precision': not_buy_precision
    }

# ---------------------------------------------------------------
# 6.  BAYESIAN OPTIMIZATION FOR XGBOOST ONLY
def train_xgboost_bayesian(X_train, X_test, y_train, y_test, buy_class, not_buy_class):
    """Train XGBoost with Bayesian optimization - get top 2 models"""

    print("\n" + "="*70)
    print("BAYESIAN OPTIMIZATION: XGBOOST (Top 2 Models)")
    print("="*70)

    param_space_xgb = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 8),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.7, 1.0),
        'colsample_bytree': Real(0.7, 1.0),
        'reg_alpha': Real(0, 1.0),
        'reg_lambda': Real(0.5, 2.0),
        'min_child_weight': Integer(1, 5)
    }

    xgb_clf = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
    xgb_search = BayesSearchCV(xgb_clf, param_space_xgb, n_iter=N_ITER, cv=3, scoring='roc_auc',
                               n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    xgb_search.fit(X_train, y_train)

    # Get top 2 XGBoost models
    xgb_results = []
    for i in range(min(2, len(xgb_search.cv_results_['params']))):
        idx = np.argsort(xgb_search.cv_results_['mean_test_score'])[::-1][i]
        params = xgb_search.cv_results_['params'][idx]
        cv_score = xgb_search.cv_results_['mean_test_score'][idx]

        model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
        model.fit(X_train, y_train)

        metrics = evaluate_model_comprehensive(model, X_test, y_test, 'XGBoost', i+1, buy_class, not_buy_class)

        xgb_results.append({
            'model': model, 'params': params, 'cv_score': cv_score, 'rank': i+1, **metrics
        })

    return xgb_results

# ---------------------------------------------------------------
# 7.  SINGLE STOCK TRAINING FUNCTION
def train_single_stock(csv_path, stock_name):
    """Train XGBoost model for a single stock"""
    print(f"\n{'='*100}")
    print(f"🚀 TRAINING MODEL FOR: {stock_name}")
    print(f"📂 File: {csv_path}")
    print(f"{'='*100}")
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = pd.read_csv(csv_path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # FILTER DATA FOR 2020-2025 ONLY
        print(f"Original dataset date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
        df = df[(df[DATE_COL] >= '2020-01-01') & (df[DATE_COL] <= '2025-12-31')]
        print(f"Filtered dataset (2020-2025) date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
        print(f"Filtered dataset size: {len(df):,} rows")

        df = df.sort_values(DATE_COL).reset_index(drop=True)

        # Convert features to numeric
        for col in FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # CRITICAL: Remove NaN rows instead of filling them
        df = clean_data_remove_nan(df, FEATURES, TARGET_COL)

        # Convert to proper data types
        for col in FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        # Target encoding
        label_encoder = LabelEncoder()
        df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL])

        # Identify classes
        class_counts = df[TARGET_COL].value_counts()
        print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, [0, 1]))}")
        print(f"Clean dataset class distribution: {dict(class_counts.sort_index())}")

        buy_class = None
        not_buy_class = None

        for encoded_val, original_label in zip([0, 1], label_encoder.classes_):
            if 'buy' in original_label.lower() and 'not' not in original_label.lower():
                buy_class = encoded_val
            else:
                not_buy_class = encoded_val

        print(f"Buy class (encoded): {buy_class} ('{label_encoder.classes_[buy_class]}')")
        print(f"Not-Buy class (encoded): {not_buy_class} ('{label_encoder.classes_[not_buy_class]}')")
        print(f"Original imbalance ratio: {class_counts[not_buy_class]/class_counts[buy_class]:.1f}:1")

        # Apply diverse sampling with 2020-2025 dataset
        balanced_df = diverse_temporal_sampling_full_dataset(
            df, TARGET_COL, buy_class, not_buy_class,
            FEATURES, buy_ratio=BUY_RATIO, n_clusters=N_CLUSTERS,
            temporal_col=DATE_COL
        )

        # *** ADD LOOKBACK FEATURES HERE (AFTER SAMPLING, BEFORE BAYESIAN OPTIMIZATION) ***
        print("\n" + "="*70)
        print("ADDING LOOKBACK FEATURES FOR ENHANCED MODEL TRAINING")
        print("="*70)
        
        balanced_df_with_lookback, extended_features = add_lookback_features(
            balanced_df, FEATURES, LOOKBACK_PERIODS, DATE_COL
        )

        # Prepare features and target with extended feature set
        X = balanced_df_with_lookback[extended_features]
        y = balanced_df_with_lookback[TARGET_COL]

        # Verify no NaN in final dataset
        final_nan_count = X.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"⚠️  Warning: {final_nan_count} NaN values found in final feature matrix!")
            # Remove any remaining NaN rows
            nan_mask = X.isnull().any(axis=1)
            X = X[~nan_mask]
            y = y[~nan_mask]
            print(f"   Removed {nan_mask.sum()} rows with NaN values")
            print(f"   Final clean dataset size: {len(X):,} rows")
        else:
            print("✅ Final dataset is NaN-free")

        # Split data with 0.2 test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nDataset sizes (with lookback features):")
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        print(f"  Feature count: {len(extended_features)} (original: {len(FEATURES)}, lookback: {len(extended_features)-len(FEATURES)})")
        print(f"  Training class distribution: {dict(pd.Series(y_train).value_counts().sort_index())}")

        # Train XGBoost with Bayesian optimization
        xgb_results = train_xgboost_bayesian(X_train_scaled, X_test_scaled, y_train, y_test, buy_class, not_buy_class)

        # Use XGBoost results only
        all_results = xgb_results

        # Sort by AUC for best models
        all_results.sort(key=lambda x: x['auc'], reverse=True)

        print(f"\n" + "="*90)
        print(f"🏆 FINAL RESULTS SUMMARY - TOP 2 XGBOOST MODELS WITH LOOKBACK FEATURES")
        print("="*90)

        print(f"\n🥇 ALL MODELS RANKED BY AUC:")
        print("-" * 90)
        for i, model in enumerate(all_results):
            print(f"{i+1}. XGBoost (Rank {model['rank']}) - AUC: {model['auc']:.4f}")

        # Save best model overall
        best_model_info = all_results[0]
        best_model = best_model_info['model']

        # Save model and params
        model_path = os.path.join(MODELS_FOLDER, f'{stock_name}_xgb.pkl')
        params_path = os.path.join(MODELS_FOLDER, f'{stock_name}_xgb_params.pkl')
        
        joblib.dump(best_model, model_path)
        joblib.dump(best_model_info['params'], params_path)

        print(f"\n💾 MODELS SAVED:")
        print("-" * 60)
        print(f"   ✅ Best XGBoost Model: {model_path} (AUC: {best_model_info['auc']:.4f})")
        print(f"   ✅ Parameters: {params_path}")

        # Return results for summary file
        return {
            'stock_name': stock_name,
            'accuracy': best_model_info['accuracy'],
            'buy_precision': best_model_info['precision'],
            'not_buy_precision': best_model_info['not_buy_precision'],
            'auc_score': best_model_info['auc'],
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"\n❌ ERROR TRAINING {stock_name}: {str(e)}")
        return {
            'stock_name': stock_name,
            'accuracy': 0.0,
            'buy_precision': 0.0,
            'not_buy_precision': 0.0,
            'auc_score': 0.0,
            'status': f'ERROR: {str(e)}'
        }

# ---------------------------------------------------------------
# 8.  MAIN EXECUTION - MULTI STOCK TRAINING
def main():
    """Train XGBoost models for all stocks in the midcap data folder"""
    print("="*100)
    print("🚀 MULTI-STOCK XGBOOST TRAINING PIPELINE")
    print("="*100)
    
    # Create models directory
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    
    # Find all CSV files in the data folder
    csv_pattern = os.path.join(DATA_FOLDER, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found in {DATA_FOLDER}")
        return
    
    print(f"📂 Found {len(csv_files)} CSV files in {DATA_FOLDER}")
    
    # Initialize results list
    all_results = []
    
    # Process each stock
    for i, csv_path in enumerate(csv_files, 1):
        # Extract stock name from filename
        filename = os.path.basename(csv_path)
        stock_name = filename.replace('_1min_data_prep.csv', '').replace('.csv', '')
        
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(csv_files)}: {stock_name}")
        print(f"{'='*50}")
        
        # Train model for this stock
        result = train_single_stock(csv_path, stock_name)
        all_results.append(result)
        
        print(f"\n✅ Completed {stock_name}")
    
    # Write results summary file
    print(f"\n{'='*100}")
    print("📝 WRITING RESULTS SUMMARY")
    print("="*100)
    
    with open(RESULTS_FILE, 'w') as f:
        # Write header
        f.write("stock_name,accuracy,buy_precision,not_buy_precision,auc_score,status\n")
        
        # Write results for each stock
        for result in all_results:
            f.write(f"{result['stock_name']},{result['accuracy']:.4f},{result['buy_precision']:.4f},"
                   f"{result['not_buy_precision']:.4f},{result['auc_score']:.4f},{result['status']}\n")
    
    print(f"✅ Results summary saved to: {RESULTS_FILE}")
    
    # Print final summary
    successful_stocks = [r for r in all_results if r['status'] == 'SUCCESS']
    failed_stocks = [r for r in all_results if r['status'] != 'SUCCESS']
    
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Total stocks processed: {len(all_results)}")
    print(f"   Successfully trained: {len(successful_stocks)}")
    print(f"   Failed: {len(failed_stocks)}")
    
    if successful_stocks:
        avg_auc = np.mean([r['auc_score'] for r in successful_stocks])
        best_stock = max(successful_stocks, key=lambda x: x['auc_score'])
        print(f"   Average AUC: {avg_auc:.4f}")
        print(f"   Best performing stock: {best_stock['stock_name']} (AUC: {best_stock['auc_score']:.4f})")
    
    if failed_stocks:
        print(f"\n❌ Failed stocks:")
        for stock in failed_stocks:
            print(f"   - {stock['stock_name']}: {stock['status']}")
    
    print(f"\n📁 All models saved in: {MODELS_FOLDER}/")
    print(f"📊 Results summary: {RESULTS_FILE}")
    
    return all_results

# Run the enhanced pipeline for all stocks
if __name__ == "__main__":
    results = main()