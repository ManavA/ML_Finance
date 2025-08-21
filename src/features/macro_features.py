#!/usr/bin/env python3
"""
Macroeconomic Feature Engineering
Integrates economic indicators with market data for enhanced ML features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Import economic indicators module
import sys
sys.path.append('../data')
from data.polygon_economic_indicators import PolygonEconomicIndicators

logger = logging.getLogger(__name__)

class MacroFeatureEngineer:
    """
    Engineer macroeconomic features for ML trading models
    Combines market data with economic indicators
    """
    
    def __init__(self, polygon_api_key: Optional[str] = None):
        """
        Initialize macro feature engineer
        
        Args:
            polygon_api_key: Polygon API key for economic data
        """
        self.econ_client = PolygonEconomicIndicators(api_key=polygon_api_key)
        
        # Feature configuration
        self.macro_features = {
            'inflation': [
                'cpi_yoy', 'core_cpi_yoy', 'pce_yoy', 
                'inflation_exp_1y', 'inflation_exp_5y', 'inflation_surprise'
            ],
            'rates': [
                'fed_funds_rate', 'real_rate_10y', 'yield_curve_2_10',
                'yield_curve_slope', 'term_premium', 'rate_volatility'
            ],
            'economic': [
                'gdp_growth_yoy', 'unemployment_rate', 'unemployment_change',
                'consumer_confidence', 'economic_surprise_index'
            ],
            'market_risk': [
                'vix_level', 'vix_regime', 'credit_spreads', 
                'dollar_index', 'volatility_of_volatility'
            ],
            'regime': [
                'monetary_policy_stance', 'inflation_regime', 
                'growth_regime', 'risk_regime', 'yield_curve_regime'
            ]
        }
        
    def create_macro_features(self,
                            market_data: pd.DataFrame,
                            start_date: str = None,
                            end_date: str = None,
                            feature_groups: List[str] = None) -> pd.DataFrame:
        """
        Create comprehensive macro features combined with market data
        
        Args:
            market_data: DataFrame with OHLCV market data
            start_date: Start date for economic data
            end_date: End date for economic data
            feature_groups: Which feature groups to include
            
        Returns:
            DataFrame with market data and macro features
        """
        logger.info("Creating macro-enhanced features...")
        
        # Get date range from market data if not specified
        if start_date is None:
            start_date = market_data.index.min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = market_data.index.max().strftime('%Y-%m-%d')
        
        # Fetch economic indicators
        macro_data = self.econ_client.create_macro_features(start_date, end_date)
        
        # Align with market data frequency
        aligned_macro = self._align_macro_data(macro_data, market_data.index)
        
        # Combine with market data
        enhanced_data = market_data.copy()
        
        # Add macro features
        feature_groups = feature_groups or ['inflation', 'rates', 'economic', 'market_risk']
        
        for group in feature_groups:
            if group == 'inflation':
                enhanced_data = self._add_inflation_features(enhanced_data, aligned_macro)
            elif group == 'rates':
                enhanced_data = self._add_rate_features(enhanced_data, aligned_macro)
            elif group == 'economic':
                enhanced_data = self._add_economic_features(enhanced_data, aligned_macro)
            elif group == 'market_risk':
                enhanced_data = self._add_risk_features(enhanced_data, aligned_macro)
            elif group == 'regime':
                enhanced_data = self._add_regime_features(enhanced_data, aligned_macro)
        
        # Create interaction features
        enhanced_data = self._create_interaction_features(enhanced_data)
        
        # Create lead/lag features for economic indicators
        enhanced_data = self._create_lead_lag_features(enhanced_data)
        
        logger.info(f"Created {len(enhanced_data.columns) - len(market_data.columns)} macro features")
        
        return enhanced_data
    
    def _align_macro_data(self, macro_data: pd.DataFrame, market_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align macro data (often monthly) with market data frequency (daily/hourly)
        """
        if macro_data.empty:
            return pd.DataFrame(index=market_index)
        
        # Reindex to market frequency and forward fill
        aligned = macro_data.reindex(market_index, method='ffill')
        
        # Handle any remaining NaN at the beginning
        aligned = aligned.fillna(method='bfill')
        
        return aligned
    
    def _add_inflation_features(self, data: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        """Add inflation-related features"""
        
        # Core inflation metrics
        if 'cpi_yoy' in macro.columns:
            data['macro_cpi_yoy'] = macro['cpi_yoy']
            data['macro_cpi_acceleration'] = macro['cpi_yoy'].diff()
            data['macro_cpi_ma3'] = macro['cpi_yoy'].rolling(3).mean()
        
        if 'core_cpi_yoy' in macro.columns:
            data['macro_core_cpi'] = macro['core_cpi_yoy']
            data['macro_cpi_core_spread'] = macro['cpi_yoy'] - macro['core_cpi_yoy']
        
        # Inflation expectations
        if 'inflation_exp_1y' in macro.columns:
            data['macro_inflation_exp_1y'] = macro['inflation_exp_1y']
        
        if 'inflation_exp_5y' in macro.columns:
            data['macro_inflation_exp_5y'] = macro['inflation_exp_5y']
            data['macro_inflation_exp_curve'] = macro['inflation_exp_5y'] - macro['inflation_exp_1y']
        
        # Inflation surprise
        if 'cpi_yoy' in macro.columns and 'inflation_exp_1y' in macro.columns:
            data['macro_inflation_surprise'] = macro['cpi_yoy'] - macro['inflation_exp_1y']
        
        # Inflation regime
        if 'cpi_yoy' in macro.columns:
            data['macro_inflation_regime'] = pd.cut(
                macro['cpi_yoy'],
                bins=[-np.inf, 2, 3, 5, np.inf],
                labels=['low', 'target', 'elevated', 'high']
            )
            
            # Convert to dummy variables
            inflation_dummies = pd.get_dummies(data['macro_inflation_regime'], prefix='inflation_regime')
            data = pd.concat([data, inflation_dummies], axis=1)
            data = data.drop('macro_inflation_regime', axis=1)
        
        return data
    
    def _add_rate_features(self, data: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        """Add interest rate features"""
        
        # Fed funds rate
        if 'effective_rate' in macro.columns:
            data['macro_fed_funds'] = macro['effective_rate']
            data['macro_fed_funds_change'] = macro['effective_rate'].diff()
            data['macro_fed_funds_ma3'] = macro['effective_rate'].rolling(3).mean()
        
        # Yield curve
        if 'yield_curve_2_10' in macro.columns:
            data['macro_yield_curve_2_10'] = macro['yield_curve_2_10']
            data['macro_yield_curve_inverted'] = (macro['yield_curve_2_10'] < 0).astype(int)
            data['macro_yield_curve_change'] = macro['yield_curve_2_10'].diff()
        
        # Real rates
        if 'real_rate_10y' in macro.columns:
            data['macro_real_rate_10y'] = macro['real_rate_10y']
            data['macro_real_rate_positive'] = (macro['real_rate_10y'] > 0).astype(int)
        
        # Term structure
        if 'yield_10Y' in macro.columns:
            data['macro_10y_yield'] = macro['yield_10Y']
            data['macro_10y_yield_ma20'] = macro['yield_10Y'].rolling(20).mean()
            
        if 'yield_2Y' in macro.columns:
            data['macro_2y_yield'] = macro['yield_2Y']
        
        # Rate volatility
        if 'yield_10Y' in macro.columns:
            data['macro_rate_volatility'] = macro['yield_10Y'].rolling(20).std()
        
        # Monetary policy stance
        if 'effective_rate' in macro.columns:
            rate_change_12m = macro['effective_rate'].diff(12)
            data['macro_fed_hiking'] = (rate_change_12m > 0.5).astype(int)
            data['macro_fed_cutting'] = (rate_change_12m < -0.5).astype(int)
            data['macro_fed_neutral'] = ((rate_change_12m >= -0.5) & (rate_change_12m <= 0.5)).astype(int)
        
        return data
    
    def _add_economic_features(self, data: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        """Add economic growth and activity features"""
        
        # GDP growth
        if 'gdp_growth_yoy' in macro.columns:
            data['macro_gdp_growth'] = macro['gdp_growth_yoy']
            data['macro_gdp_acceleration'] = macro['gdp_growth_yoy'].diff()
            data['macro_gdp_above_trend'] = (macro['gdp_growth_yoy'] > 2.5).astype(int)
        
        # Unemployment
        if 'unemployment_rate' in macro.columns:
            data['macro_unemployment'] = macro['unemployment_rate']
            data['macro_unemployment_change'] = macro['unemployment_rate'].diff()
            data['macro_unemployment_low'] = (macro['unemployment_rate'] < 4).astype(int)
        
        # Consumer confidence
        if 'consumer_confidence' in macro.columns:
            data['macro_consumer_confidence'] = macro['consumer_confidence']
            data['macro_confidence_ma3'] = macro['consumer_confidence'].rolling(3).mean()
            data['macro_confidence_change'] = macro['consumer_confidence'].diff()
        
        # Economic surprise index
        if 'gdp_growth_yoy' in macro.columns and 'unemployment_rate' in macro.columns:
            # Simple economic surprise index
            gdp_surprise = macro['gdp_growth_yoy'] - macro['gdp_growth_yoy'].rolling(12).mean()
            unemp_surprise = -(macro['unemployment_rate'] - macro['unemployment_rate'].rolling(12).mean())
            data['macro_economic_surprise'] = (gdp_surprise + unemp_surprise) / 2
        
        # Growth regime
        if 'gdp_growth_yoy' in macro.columns:
            data['macro_growth_regime'] = pd.cut(
                macro['gdp_growth_yoy'],
                bins=[-np.inf, 0, 2, 3, np.inf],
                labels=['recession', 'slow', 'moderate', 'strong']
            )
            
            # Convert to dummies
            growth_dummies = pd.get_dummies(data['macro_growth_regime'], prefix='growth_regime')
            data = pd.concat([data, growth_dummies], axis=1)
            data = data.drop('macro_growth_regime', axis=1)
        
        return data
    
    def _add_risk_features(self, data: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        """Add market risk and sentiment features"""
        
        # VIX
        if 'vix_close' in macro.columns:
            data['macro_vix'] = macro['vix_close']
            data['macro_vix_ma20'] = macro['vix_ma20']
            data['macro_vix_high'] = (macro['vix_close'] > 20).astype(int)
            data['macro_vix_extreme'] = (macro['vix_close'] > 30).astype(int)
            
            # Volatility of volatility
            data['macro_vix_volatility'] = macro['vix_close'].rolling(20).std()
        
        # Dollar index
        if 'dollar_index' in macro.columns:
            data['macro_dollar_index'] = macro['dollar_index']
            data['macro_dollar_change'] = macro['dollar_change']
            data['macro_dollar_strength'] = (macro['dollar_index'] > macro['dollar_ma50']).astype(int)
        
        # Credit spreads
        if 'ig_spread' in macro.columns:
            data['macro_ig_spread'] = macro['ig_spread']
        
        if 'hy_spread' in macro.columns:
            data['macro_hy_spread'] = macro['hy_spread']
            data['macro_credit_stress'] = (macro['hy_spread'] > 5).astype(int)
        
        # Risk regime
        if 'vix_close' in macro.columns:
            data['macro_risk_regime'] = pd.cut(
                macro['vix_close'],
                bins=[0, 12, 20, 30, 100],
                labels=['low_risk', 'normal_risk', 'elevated_risk', 'high_risk']
            )
            
            # Convert to dummies
            risk_dummies = pd.get_dummies(data['macro_risk_regime'], prefix='risk_regime')
            data = pd.concat([data, risk_dummies], axis=1)
            data = data.drop('macro_risk_regime', axis=1)
        
        return data
    
    def _add_regime_features(self, data: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        """Add regime and state features"""
        
        # Combined regime indicator
        regimes = []
        
        # Monetary policy regime
        if 'effective_rate' in macro.columns:
            rate_change = macro['effective_rate'].diff(12)
            monetary_regime = pd.cut(
                rate_change,
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=['easing', 'neutral', 'tightening']
            )
            regimes.append(monetary_regime)
        
        # Inflation regime
        if 'cpi_yoy' in macro.columns:
            inflation_regime = pd.cut(
                macro['cpi_yoy'],
                bins=[-np.inf, 2, 3, np.inf],
                labels=['low_inflation', 'target_inflation', 'high_inflation']
            )
            regimes.append(inflation_regime)
        
        # Create composite regime
        if regimes:
            # Combine regimes into single indicator
            regime_str = pd.DataFrame(regimes).apply(lambda x: '_'.join(x.astype(str)), axis=0)
            data['macro_composite_regime'] = regime_str
            
            # Create dummy variables for top regimes
            top_regimes = regime_str.value_counts().head(10).index
            for regime in top_regimes:
                data[f'regime_{regime}'] = (regime_str == regime).astype(int)
        
        # Business cycle indicator
        if 'gdp_growth_yoy' in macro.columns and 'unemployment_rate' in macro.columns:
            # Simple business cycle scoring
            gdp_score = (macro['gdp_growth_yoy'] > macro['gdp_growth_yoy'].rolling(12).mean()).astype(int)
            unemp_score = (macro['unemployment_rate'] < macro['unemployment_rate'].rolling(12).mean()).astype(int)
            
            cycle_score = gdp_score + unemp_score
            data['macro_cycle_expansion'] = (cycle_score >= 2).astype(int)
            data['macro_cycle_contraction'] = (cycle_score == 0).astype(int)
            data['macro_cycle_transition'] = (cycle_score == 1).astype(int)
        
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between macro and market data"""
        
        # Market returns vs economic conditions
        if 'returns' in data.columns:
            # Returns during different VIX regimes
            if 'macro_vix_high' in data.columns:
                data['returns_x_high_vix'] = data['returns'] * data['macro_vix_high']
            
            # Returns during monetary policy regimes
            if 'macro_fed_hiking' in data.columns:
                data['returns_x_fed_hiking'] = data['returns'] * data['macro_fed_hiking']
            
            # Returns with inflation
            if 'macro_cpi_yoy' in data.columns:
                data['real_returns'] = data['returns'] - (data['macro_cpi_yoy'] / 1200)  # Monthly inflation
        
        # Volume interactions
        if 'volume' in data.columns:
            # Volume during risk regimes
            if 'macro_vix_high' in data.columns:
                data['volume_x_risk'] = data['volume'] * data['macro_vix_high']
            
            # Volume with dollar strength
            if 'macro_dollar_strength' in data.columns:
                data['volume_x_dollar'] = data['volume'] * data['macro_dollar_strength']
        
        # Volatility interactions
        if 'volatility' in data.columns:
            # Market vol vs VIX
            if 'macro_vix' in data.columns:
                data['vol_vix_ratio'] = data['volatility'] / (data['macro_vix'] / 100 + 0.01)
            
            # Vol with rates
            if 'macro_rate_volatility' in data.columns:
                data['vol_x_rate_vol'] = data['volatility'] * data['macro_rate_volatility']
        
        return data
    
    def _create_lead_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lead and lag features for economic indicators"""
        
        # Economic indicators often lead/lag market movements
        lag_features = []
        lead_features = []
        
        # Identify macro columns
        macro_cols = [col for col in data.columns if col.startswith('macro_')]
        
        for col in macro_cols:
            # Skip categorical/dummy variables
            if data[col].dtype in ['object', 'category'] or col.endswith('_regime'):
                continue
            
            # Create lags (1, 3, 6 months)
            for lag in [20, 60, 120]:  # Approximate trading days
                data[f'{col}_lag{lag}'] = data[col].shift(lag)
                lag_features.append(f'{col}_lag{lag}')
            
            # Create leads for certain indicators (controversial but sometimes useful)
            # Only for slow-moving indicators
            if any(ind in col for ind in ['gdp', 'unemployment', 'inflation_exp']):
                data[f'{col}_lead20'] = data[col].shift(-20)
                lead_features.append(f'{col}_lead20')
            
            # Rate of change
            data[f'{col}_roc'] = data[col].pct_change(20)
        
        logger.info(f"Created {len(lag_features)} lag features and {len(lead_features)} lead features")
        
        return data
    
    def get_feature_importance_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze importance of macro features
        
        Returns DataFrame with feature statistics
        """
        macro_cols = [col for col in data.columns if 'macro_' in col or 'regime_' in col]
        
        importance_data = []
        
        for col in macro_cols:
            if data[col].dtype in ['float64', 'int64']:
                importance_data.append({
                    'feature': col,
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'missing_pct': data[col].isna().sum() / len(data) * 100,
                    'unique_values': data[col].nunique(),
                    'correlation_with_returns': data[col].corr(data['returns']) if 'returns' in data.columns else np.nan
                })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('correlation_with_returns', key=abs, ascending=False)
        
        return importance_df


def main():
    """Test macro feature engineering"""
    print("="*60)
    print("MACROECONOMIC FEATURE ENGINEERING")
    print("="*60)
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    market_data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)) * 2,
        'high': 102 + np.random.randn(len(dates)) * 2,
        'low': 98 + np.random.randn(len(dates)) * 2,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': 1000000 + np.random.randn(len(dates)) * 100000
    }, index=dates)
    
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(20).std()
    
    # Initialize macro feature engineer
    mfe = MacroFeatureEngineer()
    
    # Create macro-enhanced features
    print("\nCreating macro features...")
    enhanced_data = mfe.create_macro_features(
        market_data,
        start_date='2023-01-01',
        end_date='2024-12-31',
        feature_groups=['inflation', 'rates', 'economic', 'market_risk', 'regime']
    )
    
    print(f"\nOriginal features: {len(market_data.columns)}")
    print(f"Enhanced features: {len(enhanced_data.columns)}")
    print(f"Macro features added: {len(enhanced_data.columns) - len(market_data.columns)}")
    
    # Show feature categories
    feature_types = {
        'Inflation': [col for col in enhanced_data.columns if 'inflation' in col or 'cpi' in col],
        'Rates': [col for col in enhanced_data.columns if 'rate' in col or 'yield' in col],
        'Economic': [col for col in enhanced_data.columns if any(x in col for x in ['gdp', 'unemployment', 'confidence'])],
        'Risk': [col for col in enhanced_data.columns if any(x in col for x in ['vix', 'credit', 'risk'])],
        'Regime': [col for col in enhanced_data.columns if 'regime' in col]
    }
    
    print("\nFeature breakdown:")
    for category, features in feature_types.items():
        if features:
            print(f"\n{category} ({len(features)} features):")
            for feature in features[:5]:
                print(f"  - {feature}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance = mfe.get_feature_importance_analysis(enhanced_data)
    
    if not importance.empty:
        print("\nTop 10 most correlated macro features with returns:")
        for idx, row in importance.head(10).iterrows():
            if not pd.isna(row['correlation_with_returns']):
                print(f"  {row['feature']}: {row['correlation_with_returns']:.3f}")
    
    print("\n" + "="*60)
    print("Macro features ready for ML models!")
    print("="*60)

if __name__ == "__main__":
    main()