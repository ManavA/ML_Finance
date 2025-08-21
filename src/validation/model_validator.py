"""
Model Validation and Testing Framework
=======================================
Comprehensive validation suite for ML models with proper walk-forward
validation, statistical tests, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Implements proper walk-forward validation for time series
    Based on 2024 best practices for cryptocurrency ML
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 train_size: int = 252*2,  # 2 years of daily data
                 test_size: int = 63,       # 3 months
                 gap: int = 5):             # 5 days gap to prevent lookahead
        """
        Initialize walk-forward validator
        
        Args:
            n_splits: Number of walk-forward folds
            train_size: Training window size (days)
            test_size: Test window size (days)
            gap: Gap between train and test to prevent information leakage
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple]:
        """
        Generate train/test indices for walk-forward validation
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        n_samples = len(X)
        
        # Calculate step size for walk-forward
        step_size = self.test_size
        
        for i in range(self.n_splits):
            # Calculate train start and end
            train_start = i * step_size
            train_end = train_start + self.train_size
            
            # Calculate test start and end with gap
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            # Check if we have enough data
            if test_end > n_samples:
                break
            
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                       model_name: str = "Model") -> Dict:
        """
        Perform walk-forward validation and return comprehensive metrics
        """
        results = {
            'model_name': model_name,
            'fold_results': [],
            'predictions': [],
            'actuals': []
        }
        
        splits = self.split(X, y)
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # Get train and test data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(y_test, y_pred)
            fold_metrics['fold'] = fold_idx + 1
            
            results['fold_results'].append(fold_metrics)
            results['predictions'].extend(y_pred)
            results['actuals'].extend(y_test)
        
        # Calculate aggregate metrics
        all_actuals = np.array(results['actuals'])
        all_predictions = np.array(results['predictions'])
        results['aggregate_metrics'] = self.calculate_metrics(all_actuals, all_predictions)
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        metrics = {}
        
        # Regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = (true_direction == pred_direction).mean()
        else:
            metrics['directional_accuracy'] = 0
        
        # Trading metrics
        returns_true = np.diff(y_true) / y_true[:-1] if len(y_true) > 1 else np.array([0])
        returns_pred = np.diff(y_pred) / y_pred[:-1] if len(y_pred) > 1 else np.array([0])
        
        # Sharpe ratio of predicted returns
        if returns_pred.std() > 0:
            metrics['sharpe_ratio'] = (returns_pred.mean() / returns_pred.std()) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Hit rate (profitable predictions)
        metrics['hit_rate'] = (returns_pred > 0).mean()
        
        return metrics


class StatisticalValidator:
    """
    Statistical validation tests for model predictions
    Based on 2024 research best practices
    """
    
    def __init__(self):
        self.test_results = {}
    
    def diebold_mariano_test(self, 
                             errors1: np.ndarray, 
                             errors2: np.ndarray,
                             h: int = 1) -> Dict:
        """
        Diebold-Mariano test for comparing forecast accuracy
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            h: Forecast horizon
            
        Returns:
            Test statistics and p-value
        """
        d = errors1**2 - errors2**2  # Using squared loss
        
        # Calculate test statistic
        mean_d = np.mean(d)
        var_d = np.var(d)
        n = len(d)
        
        # Adjust variance for autocorrelation
        if h > 1:
            # Calculate autocorrelations
            autocorr = [np.corrcoef(d[:-i], d[i:])[0, 1] for i in range(1, h)]
            var_d = var_d * (1 + 2 * sum(autocorr))
        
        dm_stat = mean_d / np.sqrt(var_d / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'interpretation': 'Model 1 better' if dm_stat < 0 else 'Model 2 better'
        }
    
    def white_reality_check(self, returns: np.ndarray, n_bootstrap: int = 1000) -> Dict:
        """
        White's Reality Check for data snooping bias
        Tests if best strategy is genuinely good or just lucky
        """
        n = len(returns)
        
        # Bootstrap distribution under null hypothesis
        bootstrap_max_sharpe = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_returns = np.random.choice(returns, size=n, replace=True)
            
            # Center around zero (null hypothesis)
            bootstrap_returns = bootstrap_returns - returns.mean()
            
            # Calculate Sharpe ratio
            if bootstrap_returns.std() > 0:
                sharpe = (bootstrap_returns.mean() / bootstrap_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0
            
            bootstrap_max_sharpe.append(sharpe)
        
        # Calculate actual Sharpe ratio
        actual_sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate p-value
        p_value = (np.array(bootstrap_max_sharpe) >= actual_sharpe).mean()
        
        return {
            'actual_sharpe': actual_sharpe,
            'bootstrap_mean': np.mean(bootstrap_max_sharpe),
            'bootstrap_std': np.std(bootstrap_max_sharpe),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def variance_ratio_test(self, returns: pd.Series, lags: List[int] = [2, 4, 8, 16]) -> Dict:
        """
        Variance ratio test for market efficiency
        Tests if returns follow random walk
        """
        results = {}
        
        for lag in lags:
            # Calculate variance ratio
            ret_1 = returns
            ret_k = returns.rolling(lag).sum().dropna()
            
            var_1 = ret_1.var()
            var_k = ret_k.var() / lag
            
            vr = var_k / var_1
            
            # Test statistic (under homoscedasticity)
            n = len(returns)
            z_stat = (vr - 1) * np.sqrt(n * 2 * (lag - 1) / (3 * lag))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results[f'lag_{lag}'] = {
                'variance_ratio': vr,
                'z_statistic': z_stat,
                'p_value': p_value,
                'random_walk': p_value > 0.05
            }
        
        return results


class HyperparameterValidator:
    """
    Validates hyperparameter choices based on 2024 best practices
    """
    
    @staticmethod
    def get_xgboost_params(data_size: int, task: str = 'regression') -> Dict:
        """
        Get recommended XGBoost parameters based on data size
        Based on 2024 cryptocurrency prediction research
        """
        if data_size < 10000:
            # Small dataset - prevent overfitting
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        elif data_size < 100000:
            # Medium dataset
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.05,
                'reg_alpha': 0.05,
                'reg_lambda': 1.0
            }
        else:
            # Large dataset
            params = {
                'n_estimators': 300,
                'max_depth': 7,
                'learning_rate': 0.01,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1.0
            }
        
        if task == 'regression':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        
        return params
    
    @staticmethod
    def get_lightgbm_params(data_size: int, task: str = 'regression') -> Dict:
        """
        Get recommended LightGBM parameters based on data size
        Based on 2024 research showing LightGBM efficiency
        """
        if data_size < 10000:
            # Small dataset
            params = {
                'n_estimators': 100,
                'num_leaves': 15,  # Less than 2^max_depth
                'max_depth': 4,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
        elif data_size < 100000:
            # Medium dataset
            params = {
                'n_estimators': 200,
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.05,
                'lambda_l2': 0.05
            }
        else:
            # Large dataset - LightGBM excels here
            params = {
                'n_estimators': 500,
                'num_leaves': 63,
                'max_depth': 8,
                'learning_rate': 0.01,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'min_data_in_leaf': 50,
                'lambda_l1': 0,
                'lambda_l2': 0
            }
        
        if task == 'regression':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        
        params['boosting_type'] = 'gbdt'
        params['verbose'] = -1
        
        return params
    
    @staticmethod
    def validate_train_test_split(train_size: int, test_size: int, 
                                 total_size: int) -> Dict:
        """
        Validate train/test split based on 2024 best practices
        """
        train_ratio = train_size / total_size
        test_ratio = test_size / total_size
        
        recommendations = []
        warnings = []
        
        # Check ratios (50% train, 25% validation, 25% test is recommended)
        if train_ratio < 0.5:
            warnings.append("Training set too small (< 50%), may lead to underfitting")
        elif train_ratio > 0.8:
            warnings.append("Training set too large (> 80%), may lead to overfitting")
        
        if test_ratio < 0.2:
            warnings.append("Test set too small (< 20%), may not be representative")
        
        # Check absolute sizes
        if train_size < 252:  # Less than 1 year of daily data
            warnings.append("Training set < 1 year, may not capture seasonality")
        
        if test_size < 63:  # Less than 3 months
            warnings.append("Test set < 3 months, may not be reliable")
        
        # Recommendations
        if not warnings:
            recommendations.append("Train/test split looks good")
        
        recommendations.append(f"Consider walk-forward validation with {max(3, total_size // 500)} folds")
        
        return {
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'warnings': warnings,
            'recommendations': recommendations,
            'valid': len(warnings) == 0
        }


class ModelDiagnostics:
    """
    Comprehensive model diagnostics and health checks
    """
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def check_overfitting(self) -> Dict:
        """Check for overfitting by comparing train and test performance"""
        
        # Get predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        # Calculate overfitting score
        rmse_ratio = test_rmse / train_rmse
        r2_diff = train_r2 - test_r2
        
        # Determine severity
        if rmse_ratio > 1.5 or r2_diff > 0.2:
            severity = "HIGH"
            recommendation = "Model is severely overfitting. Reduce complexity or add regularization."
        elif rmse_ratio > 1.2 or r2_diff > 0.1:
            severity = "MEDIUM"
            recommendation = "Some overfitting detected. Consider adding regularization."
        else:
            severity = "LOW"
            recommendation = "Model generalization looks good."
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'rmse_ratio': rmse_ratio,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_difference': r2_diff,
            'overfitting_severity': severity,
            'recommendation': recommendation
        }
    
    def residual_analysis(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """Analyze prediction residuals for patterns"""
        
        residuals = actuals - predictions
        
        # Normality test
        _, normality_p = stats.normaltest(residuals)
        
        # Autocorrelation (Ljung-Box test)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Heteroscedasticity (Breusch-Pagan test)
        _, bp_p_value = stats.levene(residuals[:len(residuals)//2], 
                                     residuals[len(residuals)//2:])
        
        return {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'normality_p_value': normality_p,
            'residuals_normal': normality_p > 0.05,
            'autocorrelation_p': lb_test['lb_pvalue'].mean(),
            'residuals_independent': lb_test['lb_pvalue'].mean() > 0.05,
            'heteroscedasticity_p': bp_p_value,
            'constant_variance': bp_p_value > 0.05
        }


# Example usage and testing
if __name__ == "__main__":
    print("Model Validation Framework Test")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    
    # Test walk-forward validation
    validator = WalkForwardValidator(n_splits=5)
    splits = validator.split(X, y)
    
    print(f"\nWalk-Forward Validation Splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")
    
    # Test hyperparameter recommendations
    hp_validator = HyperparameterValidator()
    
    xgb_params = hp_validator.get_xgboost_params(len(X))
    print(f"\nRecommended XGBoost parameters for {len(X)} samples:")
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
    
    lgb_params = hp_validator.get_lightgbm_params(len(X))
    print(f"\nRecommended LightGBM parameters for {len(X)} samples:")
    for key, value in lgb_params.items():
        print(f"  {key}: {value}")
    
    # Test train/test split validation
    split_validation = hp_validator.validate_train_test_split(700, 300, 1000)
    print(f"\nTrain/Test Split Validation:")
    print(f"  Valid: {split_validation['valid']}")
    if split_validation['warnings']:
        print("  Warnings:")
        for warning in split_validation['warnings']:
            print(f"    - {warning}")
    print("  Recommendations:")
    for rec in split_validation['recommendations']:
        print(f"    - {rec}")
    
    print("\nValidation framework ready for use!")