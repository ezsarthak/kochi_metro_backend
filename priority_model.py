import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import warnings
import os
import sys
from datetime import datetime
import json
import pickle
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

POSITIVE_FEATURES = ["fitness", "critical", "noncritical", "revenue", "penalty", "performance", "efficiency_score", "utilization", "passenger_load", "service_quality"]
NEGATIVE_FEATURES = ["maintenance", "cleaning", "clearance", "breakdowns", "efficiency", "downtime", "repair_cost", "delay_minutes", "complaints", "fuel_consumption"]

FEATURE_WEIGHTS = {
    "fitness": 1.0,
    "critical": 2.0,
    "noncritical": 1.0,
    "revenue": 1.0,
    "penalty": 1.0,
    "maintenance": -1.0,
    "cleaning": -1.0,
    "clearance": -1.0,
    "breakdowns": -1.0,
    "efficiency": -1.0,
    "performance": 1.5,
    "efficiency_score": 1.2,
    "utilization": 1.3,
    "passenger_load": 0.8,
    "service_quality": 1.4,
    "downtime": -1.5,
    "repair_cost": -1.2,
    "delay_minutes": -1.8,
    "complaints": -1.1,
    "fuel_consumption": -0.9
}

PRIORITY_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.5,
    'low': 0.2
}

class DataValidator:
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataframe(self, df, sheet_name):
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        if df.empty:
            results['is_valid'] = False
            results['issues'].append('DataFrame is empty')
            return results
        
        if df.shape[0] < 2:
            results['warnings'].append('Less than 2 rows of data')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            results['is_valid'] = False
            results['issues'].append('No numeric columns found')
        
        null_percentage = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentage[null_percentage > 50].index.tolist()
        if high_null_cols:
            results['warnings'].append(f'High null percentage in columns: {high_null_cols}')
        
        self.validation_results[sheet_name] = results
        return results
    
    def get_summary(self):
        return self.validation_results

class FeatureMatcher:
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights
        self.matched_features = {}
    
    def find_matching_columns(self, df_columns, target_features):
        matches = {}
        for feature in target_features:
            feature_matches = []
            for col in df_columns:
                col_lower = col.lower().replace('_', '').replace('-', '').replace(' ', '')
                feature_lower = feature.lower().replace('_', '').replace('-', '').replace(' ', '')
                if feature_lower in col_lower or col_lower in feature_lower:
                    feature_matches.append(col)
                elif any(keyword in col_lower for keyword in feature_lower.split()):
                    feature_matches.append(col)
            if feature_matches:
                matches[feature] = list(set(feature_matches))
        return matches
    
    def get_feature_importance_score(self, matched_features):
        total_weight = 0
        for feature, columns in matched_features.items():
            if feature in self.feature_weights:
                total_weight += abs(self.feature_weights[feature]) * len(columns)
        return total_weight

class DataPreprocessor:
    def __init__(self, scaler_type='minmax'):
        self.scaler_type = scaler_type
        self.scalers = {}
        
    def get_scaler(self, scaler_type):
        if scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            return MinMaxScaler()
    
    def preprocess_data(self, df, sheet_name):
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df_clean, []
        
        for col in numeric_cols:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        scaler = self.get_scaler(self.scaler_type)
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
        self.scalers[sheet_name] = scaler
        
        return df_clean, numeric_cols.tolist()

def compute_priority_advanced(df, feature_matcher, preprocessor, sheet_name):
    if df.empty:
        return np.array([])
    
    df_processed, numeric_cols = preprocessor.preprocess_data(df, sheet_name)
    
    if len(numeric_cols) == 0:
        return np.zeros(len(df_processed))
    
    matched_features = feature_matcher.find_matching_columns(df_processed.columns, feature_matcher.feature_weights.keys())
    feature_matcher.matched_features[sheet_name] = matched_features
    
    score = np.zeros(len(df_processed))
    feature_contributions = {}
    
    for feature, weight in feature_matcher.feature_weights.items():
        if feature in matched_features:
            feature_score = np.zeros(len(df_processed))
            for col in matched_features[feature]:
                col_contribution = weight * df_processed[col].values
                feature_score += col_contribution
                score += col_contribution
            feature_contributions[feature] = feature_score
    
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    
    return score

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_performances = {}
        
    def initialize_models(self):
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=1.0, random_state=42),
            "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
            "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
        }
        
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(random_state=42, n_estimators=100, verbosity=0, learning_rate=0.1)
            
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(random_state=42, n_estimators=100, verbose=-1, learning_rate=0.1)
            
        return models
    
    def evaluate_model(self, model, X, y):
        predictions = model.predict(X)
        
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        spearman_corr, spearman_p = spearmanr(y, predictions)
        pearson_corr, pearson_p = pearsonr(y, predictions)
        
        if np.isnan(spearman_corr):
            spearman_corr = 0
        if np.isnan(pearson_corr):
            pearson_corr = 0
            
        combined_score = (r2 + spearman_corr + pearson_corr) / 3
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'spearman': spearman_corr,
            'pearson': pearson_corr,
            'combined_score': combined_score,
            'predictions': predictions
        }
    
    def train_models_advanced(self, X, y, sheet_name):
        if X.empty or len(X) < 2:
            return None, None
            
        if np.std(y) < 1e-6:
            return None, None
        
        models = self.initialize_models()
        best_model, best_score, best_name, best_metrics = None, -np.inf, None, None
        
        model_results = {}
        
        for name, model in models.items():
            try:
                model.fit(X, y)
                metrics = self.evaluate_model(model, X, y)
                # Convert numpy arrays to lists for JSON serialization
                metrics_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
                model_results[name] = metrics_serializable
                
                if metrics['combined_score'] > best_score:
                    best_model = model
                    best_score = metrics['combined_score']
                    best_name = name
                    best_metrics = metrics
                    
            except Exception as e:
                model_results[name] = {'error': str(e)}
        
        self.model_performances[sheet_name] = model_results
        
        if best_score < 0.1:
            return None, None
            
        return best_model, best_metrics

class PriorityAssigner:
    def __init__(self):
        self.priority_distributions = {}
        
    def assign_priorities_to_25_levels(self, scores):
        n_trains = len(scores)
        
        if n_trains <= 25:
            sorted_indices = scores.argsort()[::-1]
            priorities = np.zeros(n_trains, dtype=int)
            
            for i, idx in enumerate(sorted_indices):
                priorities[idx] = i + 1
                
            return pd.Series(priorities, index=scores.index)
        else:
            percentiles = np.linspace(0, 100, 26)
            priorities = pd.cut(
                scores.rank(ascending=False, pct=True) * 100,
                bins=percentiles[::-1],
                labels=range(1, 26),
                include_lowest=True
            ).astype(int)
            return priorities
    
    def assign_priorities_advanced(self, scores, n_levels=25, method='quantile'):
        priorities = self.assign_priorities_to_25_levels(scores)
        
        distribution = priorities.value_counts().sort_index()
        self.priority_distributions['final'] = distribution.to_dict()
        
        return priorities
    
    def assign_priorities_percentile(self, scores, n_levels):
        percentiles = np.linspace(0, 100, n_levels + 1)
        priorities = pd.cut(
            scores.rank(ascending=False, pct=True) * 100,
            bins=percentiles[::-1],
            labels=range(1, n_levels + 1),
            include_lowest=True
        ).astype(int)
        return priorities
    
    def assign_priorities_threshold(self, scores, n_levels):
        thresholds = np.linspace(scores.min(), scores.max(), n_levels + 1)
        priorities = pd.cut(
            scores,
            bins=thresholds,
            labels=range(n_levels, 0, -1),
            include_lowest=True
        ).astype(int)
        return priorities

def generate_comprehensive_json_output(combined_df, original_data, validator, feature_matcher, model_trainer, priority_assigner, processing_time, valid_sheets):
    """Generate comprehensive JSON output with all train details"""
    
    # Prepare train details
    train_details = []
    for idx, row in combined_df.iterrows():
        train_info = {
            "train_id": str(idx),
            "priority_rank": int(row["final_priority"]),
            "final_score": float(row["final_score"]),
            "weighted_score": float(row["weighted_score"]),
            "scores_by_sheet": {}
        }
        
        # Add scores from each sheet
        for sheet in valid_sheets:
            if sheet in row and pd.notna(row[sheet]):
                train_info["scores_by_sheet"][sheet] = float(row[sheet])
        
        # Add original data if available
        train_info["original_data"] = {}
        for sheet_name, sheet_df in original_data.items():
            if int(idx) < len(sheet_df):
                train_info["original_data"][sheet_name] = {}
                for col, val in sheet_df.iloc[int(idx)].items():
                    if pd.isna(val):
                        train_info["original_data"][sheet_name][str(col)] = None
                    elif isinstance(val, (np.integer, int)):
                        train_info["original_data"][sheet_name][str(col)] = int(val)
                    elif isinstance(val, (np.floating, float)):
                        train_info["original_data"][sheet_name][str(col)] = float(val)
                    else:
                        train_info["original_data"][sheet_name][str(col)] = str(val)
        
        # If no original rows were available in any sheet for this train, drop the empty key
        if not train_info["original_data"]:
            del train_info["original_data"]

        train_details.append(train_info)
    
    # Sort by priority rank
    train_details.sort(key=lambda x: x["priority_rank"])
    
    # Priority distribution
    priority_distribution = {}
    for priority in range(1, 26):
        count = sum(1 for train in train_details if train["priority_rank"] == priority)
        if count > 0:
            priority_distribution[str(priority)] = count
    
    # Comprehensive output structure
    output = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "file_processed": "Final kochi 1.xlsx",  # You can make this dynamic
            "total_trains": len(combined_df),
            "total_sheets_processed": len(valid_sheets),
            "sheets_processed": valid_sheets
        },
        
        "train_priorities": train_details,
        
        "priority_statistics": {
            "distribution": priority_distribution,
            "summary": {
                "highest_priority": 1,
                "lowest_priority": max([train["priority_rank"] for train in train_details]),
                "mean_score": float(combined_df["final_score"].mean()),
                "std_score": float(combined_df["final_score"].std()),
                "min_score": float(combined_df["final_score"].min()),
                "max_score": float(combined_df["final_score"].max())
            }
        },
        
        "analysis_details": {
            "validation_summary": validator.get_summary(),
            "feature_matching": feature_matcher.matched_features,
            "model_performance": model_trainer.model_performances,
            "feature_weights_used": FEATURE_WEIGHTS,
            "priority_thresholds": PRIORITY_THRESHOLDS
        },
        
        "summary_by_priority_level": {
            "high_priority": [train for train in train_details if train["priority_rank"] <= 5],
            "medium_priority": [train for train in train_details if 6 <= train["priority_rank"] <= 15],
            "low_priority": [train for train in train_details if train["priority_rank"] >= 16]
        }
    }
    
    return output

def main_advanced_pipeline(file_path):
    start_time = datetime.now()
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    
    try:
        xl = pd.ExcelFile(file_path)
        print(f"Processing file: {file_path}")
        print(f"Found sheets: {xl.sheet_names}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    validator = DataValidator()
    feature_matcher = FeatureMatcher(FEATURE_WEIGHTS)
    preprocessor = DataPreprocessor(scaler_type='minmax')
    model_trainer = ModelTrainer()
    priority_assigner = PriorityAssigner()
    
    sheet_scores = {}
    sheet_data = {}
    original_data = {}
    valid_sheets = []

    for sheet in xl.sheet_names:
        print(f"\n{'='*60}")
        print(f"Processing sheet: {sheet}")
        print('='*60)
        
        try:
            df = xl.parse(sheet)
            original_data[sheet] = df.copy()  # Store original data
            
            validation_result = validator.validate_dataframe(df, sheet)
            
            if not validation_result['is_valid']:
                print(f"Skipping sheet '{sheet}' due to validation issues: {validation_result['issues']}")
                continue
            
            if validation_result['warnings']:
                print(f"Warnings for sheet '{sheet}': {validation_result['warnings']}")
            
            print(f"Sheet dimensions: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            y = compute_priority_advanced(df, feature_matcher, preprocessor, sheet)
            if len(y) == 0:
                continue
                
            X = df.select_dtypes(include=[np.number])
            if X.shape[1] == 0:
                print("No numeric features available for ML training")
                preds = y
                model_metrics = None
            else:
                print(f"Training features: {list(X.columns)}")
                best_model, model_metrics = model_trainer.train_models_advanced(X, y, sheet)
                if best_model is None:
                    preds = y
                    print("Using rule-based scoring due to poor model performance")
                else:
                    preds = best_model.predict(X)
                    print(f"Using ML model predictions (R²: {model_metrics['r2']:.3f})")

            df["priority_score"] = preds
            sheet_scores[sheet] = preds
            sheet_data[sheet] = df
            valid_sheets.append(sheet)
            
        except Exception as e:
            print(f"Error processing sheet '{sheet}': {e}")
            continue

    if not sheet_scores:
        print("No valid data found in any sheet")
        return None

    print(f"\n{'='*60}")
    print("COMBINING RESULTS FROM ALL SHEETS")
    print('='*60)
    
    combined_df = pd.DataFrame(sheet_scores)
    combined_df = combined_df.fillna(combined_df.mean(axis=1))
    
    weights = {}
    for sheet in valid_sheets:
        importance_score = feature_matcher.get_feature_importance_score(
            feature_matcher.matched_features.get(sheet, {})
        )
        weights[sheet] = max(importance_score, 1.0)
    
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    combined_df["weighted_score"] = sum(
        combined_df[sheet] * normalized_weights.get(sheet, 1.0) 
        for sheet in combined_df.columns if sheet in normalized_weights
    )
    
    combined_df["final_score"] = (combined_df["weighted_score"] + combined_df.mean(axis=1)) / 2
    combined_df["final_priority"] = priority_assigner.assign_priorities_advanced(
        combined_df["final_score"], 
        n_levels=25,
        method='quantile'
    )

    print("\n🚇 Final Train Priorities:")
    result_summary = combined_df[["final_score", "final_priority"]].sort_values("final_priority")
    result_summary.index.name = "Train_ID"
    print(result_summary)

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Generate comprehensive JSON output
    json_output = generate_comprehensive_json_output(
        combined_df, original_data, validator, feature_matcher, 
        model_trainer, priority_assigner, processing_time, valid_sheets
    )
    
    # Save JSON output with constant filename
    output_file = "train_priorities_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    # Also save a simplified priorities-only file for backward compatibility
    priorities_only = [train["priority_rank"] for train in json_output["train_priorities"]]
    simple_output_file = "priorities.txt"
    with open(simple_output_file, 'w') as f:
        for priority in priorities_only:
            f.write(f"{priority}\n")
    
    # Print console output about generated files
    print(f"\n{'='*60}")
    print("FILES GENERATED:")
    print('='*60)
    print(f"1. {output_file}")
    print(f"   - Complete JSON analysis with all train details")
    print(f"   - Contains: Train IDs, priorities, scores, original data")
    print(f"   - Size: {len(combined_df)} trains analyzed")
    print(f"   - Priority range: 1 to {max([train['priority_rank'] for train in json_output['train_priorities']])}")
    print()
    print(f"2. {simple_output_file}")
    print(f"   - Simple text file with priority rankings only")
    print(f"   - One priority number per line")
    print(f"   - Total entries: {len(priorities_only)}")
    print()
    print(f"Processing completed in {processing_time:.2f} seconds")
    print('='*60)

    return json_output

if __name__ == "__main__":
    file_path = "Final kochi 1.xlsx"
    
    try:
        result = main_advanced_pipeline(file_path)
        if result is not None:
            print(f"\n✅ Processing completed successfully!")
            print(f"🔍 Check the generated JSON file for complete details including:")
            print("   • Train IDs with priority rankings")
            print("   • Individual and weighted scores")
            print("   • Original data for each train")
            print("   • Analysis metadata and performance metrics")
            print("   • Priority distribution statistics")
        else:
            print("❌ Processing failed. Please check the input file and try again.")
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")
        sys.exit(1)
