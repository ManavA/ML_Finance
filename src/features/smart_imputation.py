#!/usr/bin/env python3
"""
Smart imputation strategies for financial ML features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SmartImputer:

    def __init__(self):
        self.imputation_stats = {}
        
    def impute_features(self, df: pd.DataFrame, feature_metadata: Dict = None) -> pd.DataFrame:
        
        df = df.copy()
        
        # Classify features if not provided
        if feature_metadata is None:
            feature_metadata = self._classify_features(df)
        
        # Track missingness before imputation
        missing_indicators = self._create_missing_indicators(df, feature_metadata)
        
        # Apply feature-specific imputation
        for feature_type, columns in feature_metadata.items():
            if feature_type == 'price':
                df = self._impute_price_features(df, columns)
            elif feature_type == 'volume':
                df = self._impute_volume_features(df, columns)
            elif feature_type == 'technical':
                df = self._impute_technical_features(df, columns)
            elif feature_type == 'macro':
                df = self._impute_macro_features(df, columns)
            elif feature_type == 'volatility':
                df = self._impute_volatility_features(df, columns)
            elif feature_type == 'regime':
                df = self._impute_regime_features(df, columns)
        
        # Add missing indicators as features
        df = pd.concat([df, missing_indicators], axis=1)
        
        # Log imputation statistics
        self._log_imputation_stats(df)
        
        return df
    
    def _classify_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:

        feature_types = {
            'price': [],
            'volume': [],
            'technical': [],
            'macro': [],
            'volatility': [],
            'regime': [],
            'target': [],
            'other': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(x in col_lower for x in ['open', 'high', 'low', 'close', 'price', 'return']):
                feature_types['price'].append(col)
            elif 'volume' in col_lower or 'obv' in col_lower or 'vwap' in col_lower:
                feature_types['volume'].append(col)
            elif any(x in col_lower for x in ['rsi', 'macd', 'bb_', 'atr', 'adx', 'cci', 'stoch', 'ema', 'sma']):
                feature_types['technical'].append(col)
            elif 'macro_' in col_lower or any(x in col_lower for x in ['cpi', 'gdp', 'unemployment', 'fed']):
                feature_types['macro'].append(col)
            elif 'volatility' in col_lower or 'std' in col_lower:
                feature_types['volatility'].append(col)
            elif 'regime' in col_lower:
                feature_types['regime'].append(col)
            elif 'target' in col_lower:
                feature_types['target'].append(col)
            else:
                feature_types['other'].append(col)
        
        return feature_types
    
    def _create_missing_indicators(self, df: pd.DataFrame, feature_metadata: Dict) -> pd.DataFrame:

        missing_indicators = pd.DataFrame(index=df.index)
        
        # Indicator for any missing macro data
        if 'macro' in feature_metadata and feature_metadata['macro']:
            macro_cols = feature_metadata['macro']
            missing_indicators['has_macro_update'] = (~df[macro_cols].isna().all(axis=1)).astype(int)
            
            # Days since last macro update (approximation)
            last_update = df[macro_cols].notna().any(axis=1)
            days_since = last_update.groupby((~last_update).cumsum()).cumcount()
            missing_indicators['days_since_macro'] = days_since
        
        # Indicator for technical indicator readiness
        if 'technical' in feature_metadata and feature_metadata['technical']:
            tech_cols = feature_metadata['technical']
            missing_indicators['tech_indicators_ready'] = (~df[tech_cols].isna().any(axis=1)).astype(int)
        
        # Market data quality indicator
        if 'price' in feature_metadata:
            price_cols = feature_metadata['price']
            missing_indicators['complete_market_data'] = (~df[price_cols].isna().any(axis=1)).astype(int)
        
        return missing_indicators
    
    def _impute_price_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        for col in columns:
            if col in df.columns:
                if df[col].isna().any():
                    # Forward fill for small gaps
                    df[col] = df[col].fillna(method='ffill', limit=5)
                    # Backward fill for any remaining
                    df[col] = df[col].fillna(method='bfill', limit=5)
                    # If still missing, use mean
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _impute_volume_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        for col in columns:
            if col in df.columns:
                if df[col].isna().any():
                    # Use rolling median for volume
                    rolling_median = df[col].rolling(window=24, min_periods=1).median()
                    df[col] = df[col].fillna(rolling_median)
                    # Final fallback to overall median
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _impute_technical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        for col in columns:
            if col in df.columns:
                # Identify warm-up period (first non-NaN)
                first_valid = df[col].first_valid_index()
                if first_valid is not None:
                    # Only forward fill after warm-up
                    df.loc[first_valid:, col] = df.loc[first_valid:, col].fillna(method='ffill')
                    
                    # Do NOT fill warm-up period - these NaN are meaningful
                    # Models like XGBoost handle them naturally
        
        return df
    
    def _impute_macro_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        for col in columns:
            if col in df.columns:
                if df[col].isna().any():
                    # Forward fill without limit - macro data persists
                    df[col] = df[col].fillna(method='ffill')
                    # Backward fill for initial periods
                    df[col] = df[col].fillna(method='bfill')
                    
                    # If still missing (all NaN), use 0 or neutral value
                    if df[col].isna().all():
                        # For rates/percentages, use 2% (neutral)
                        if any(x in col.lower() for x in ['rate', 'yield', 'cpi', 'inflation']):
                            df[col] = df[col].fillna(2.0)
                        # For binary indicators, use 0
                        elif df[col].dtype in ['int64', 'bool']:
                            df[col] = df[col].fillna(0)
                        # For others, use 0
                        else:
                            df[col] = df[col].fillna(0)
        
        return df
    
    def _impute_volatility_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        for col in columns:
            if col in df.columns:
                if df[col].isna().any():
                    # Use expanding volatility for early periods
                    if 'returns' in df.columns:
                        expanding_vol = df['returns'].expanding(min_periods=2).std()
                        df[col] = df[col].fillna(expanding_vol)
                    else:
                        # Forward fill
                        df[col] = df[col].fillna(method='ffill')
                        # Use median volatility as fallback
                        df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _impute_regime_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:

        for col in columns:
            if col in df.columns:
                if df[col].isna().any():
                    if df[col].dtype in ['object', 'category']:
                        # Use mode for categorical
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                        df[col] = df[col].fillna(mode_val)
                    else:
                        # Forward fill for numeric regime indicators
                        df[col] = df[col].fillna(method='ffill')
                        df[col] = df[col].fillna(0)  # Neutral regime
        
        return df
    
    def _log_imputation_stats(self, df: pd.DataFrame):

        total_missing = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        
        logger.info(f"Imputation complete:")
        logger.info(f"  Total cells: {total_cells:,}")
        logger.info(f"  Missing after imputation: {total_missing:,} ({total_missing/total_cells*100:.2f}%)")
        
        # Log problematic columns
        missing_by_col = df.isna().sum()
        problematic = missing_by_col[missing_by_col > df.shape[0] * 0.5]
        
        if len(problematic) > 0:
            logger.warning(f"Columns with >50% missing after imputation:")
            for col, missing in problematic.items():
                logger.warning(f"  {col}: {missing}/{df.shape[0]} ({missing/df.shape[0]*100:.1f}%)")
    
    def get_imputation_report(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:

        report = []
        
        for col in df_before.columns:
            if col in df_after.columns:
                before_missing = df_before[col].isna().sum()
                after_missing = df_after[col].isna().sum()
                
                if before_missing > 0:
                    report.append({
                        'feature': col,
                        'missing_before': before_missing,
                        'missing_after': after_missing,
                        'imputed': before_missing - after_missing,
                        'imputation_rate': (before_missing - after_missing) / before_missing * 100
                    })
        
        return pd.DataFrame(report).sort_values('missing_before', ascending=False)


# Model-specific handling
class MissingValueHandler:

    @staticmethod
    def prepare_for_xgboost(X: pd.DataFrame) -> pd.DataFrame:

        X_prepared = X.copy()
        
        # Convert categoricals to numeric
        for col in X_prepared.select_dtypes(include=['object', 'category']).columns:
            X_prepared[col] = pd.Categorical(X_prepared[col]).codes
        
        # XGBoost will handle NaN values automatically
        return X_prepared
    
    @staticmethod
    def prepare_for_sklearn(X: pd.DataFrame, imputer_type: str = 'smart') -> pd.DataFrame:

        if imputer_type == 'smart':
            imputer = SmartImputer()
            X_prepared = imputer.impute_features(X)
        elif imputer_type == 'simple':
            # Simple median imputation
            X_prepared = X.fillna(X.median())
        else:
            raise ValueError(f"Unknown imputer type: {imputer_type}")
        
        # Ensure no remaining NaN
        if X_prepared.isna().any().any():
            # Final fallback - fill with 0
            X_prepared = X_prepared.fillna(0)
        
        return X_prepared
    
    @staticmethod
    def prepare_for_neural_net(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
 
        # Create missingness mask
        missing_mask = X.isna().astype(float)
        
        # Smart imputation
        imputer = SmartImputer()
        X_imputed = imputer.impute_features(X)
        
        # Ensure no NaN remains
        X_imputed = X_imputed.fillna(0)
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        return X_scaled, missing_mask


def demonstrate_smart_imputation():

    print("="*60)
    print("SMART IMPUTATION STRATEGIES")
    print("="*60)
    
    # Create sample data with realistic missing patterns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Market data - mostly complete
    market_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'volume': np.random.uniform(1000, 5000, 1000),
        'returns': np.random.randn(1000) * 0.02
    }, index=dates)
    
    # Technical indicators - NaN during warm-up
    market_data['rsi'] = np.nan
    market_data.loc[dates[14:], 'rsi'] = np.random.uniform(30, 70, len(dates)-14)
    
    market_data['macd'] = np.nan
    market_data.loc[dates[26:], 'macd'] = np.random.randn(len(dates)-26) * 0.5
    
    # Macro data - monthly updates
    market_data['macro_cpi'] = np.nan
    for i in range(0, len(dates), 24*30):  # Monthly
        if i < len(dates):
            market_data.loc[dates[i], 'macro_cpi'] = 2 + np.random.randn() * 0.5
    
    market_data['macro_gdp'] = np.nan
    for i in range(0, len(dates), 24*90):  # Quarterly
        if i < len(dates):
            market_data.loc[dates[i], 'macro_gdp'] = 2.5 + np.random.randn() * 1
    
    print("\n1. Original Data Missing Pattern:")
    print(f"{'Feature':<20} {'Missing %':<12} {'Pattern'}")
    print("-"*50)
    for col in market_data.columns:
        missing_pct = market_data[col].isna().sum() / len(market_data) * 100
        print(f"{col:<20} {missing_pct:>10.1f}%  ", end="")
        
        if missing_pct == 0:
            print("Complete")
        elif missing_pct < 5:
            print("Warm-up period")
        elif missing_pct > 95:
            print("Sparse (monthly/quarterly)")
        else:
            print("Irregular")
    
    # Apply smart imputation
    print("\n2. Applying Smart Imputation...")
    imputer = SmartImputer()
    imputed_data = imputer.impute_features(market_data)
    
    print("\n3. After Smart Imputation:")
    print(f"{'Feature':<20} {'Missing %':<12} {'Imputation Method'}")
    print("-"*60)
    for col in imputed_data.columns:
        if col in market_data.columns:
            missing_pct = imputed_data[col].isna().sum() / len(imputed_data) * 100
            
            method = "None needed"
            if 'macro' in col:
                method = "Forward fill"
            elif col in ['rsi', 'macd']:
                method = "Keep warm-up NaN"
            elif col in ['close', 'volume', 'returns']:
                method = "Already complete"
            
            print(f"{col:<20} {missing_pct:>10.1f}%   {method}")
    
    # Show new indicator features
    print("\n4. New Missingness Indicator Features:")
    indicator_cols = [col for col in imputed_data.columns if col not in market_data.columns]
    for col in indicator_cols:
        print(f"  - {col}: {imputed_data[col].describe().iloc[1]:.2f} (mean)")
    
    # Demonstrate model-specific preparation
    print("\n5. Model-Specific Preparation:")
    
    # XGBoost
    X_xgb = MissingValueHandler.prepare_for_xgboost(market_data)
    print(f"\nXGBoost preparation:")
    print(f"  - Keeps NaN values: {X_xgb.isna().sum().sum()} missing cells")
    print(f"  - XGBoost handles missing naturally")
    
    # SKLearn
    X_sklearn = MissingValueHandler.prepare_for_sklearn(market_data)
    print(f"\nSKLearn preparation:")
    print(f"  - All NaN imputed: {X_sklearn.isna().sum().sum()} missing cells")
    
    # Neural Network
    X_nn, mask = MissingValueHandler.prepare_for_neural_net(market_data)
    print(f"\nNeural Network preparation:")
    print(f"  - Data normalized and imputed")
    print(f"  - Missingness mask shape: {mask.shape}")



if __name__ == "__main__":
    demonstrate_smart_imputation()