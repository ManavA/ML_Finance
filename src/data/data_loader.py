# src/data/data_loader.py
"""
Data loading utilities for ML Finance project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and caching of financial data for analysis."""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "data/cache"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
            cache_dir: Directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if it exists.
        
        Args:
            cache_key: Unique identifier for cached data
            
        Returns:
            DataFrame if cached data exists, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                logger.info(f"Loading cached data: {cache_key}")
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
                return None
        
        return None
    
    def save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """
        Save data to cache.
        
        Args:
            data: DataFrame to cache
            cache_key: Unique identifier for cached data
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            data.to_pickle(cache_file)
            logger.info(f"Saved data to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to save cache {cache_key}: {e}")
    
    def load_symbol_data(self, symbol: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a specific symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with symbol data
        """
        # Look for various file formats
        possible_files = [
            self.data_dir / f"{symbol}.pkl",
            self.data_dir / f"{symbol}.parquet",
            self.data_dir / f"{symbol}.csv",
            self.data_dir / "raw" / f"{symbol}.pkl",
            self.data_dir / "processed" / f"{symbol}.pkl"
        ]
        
        df = None
        for file_path in possible_files:
            if file_path.exists():
                try:
                    if file_path.suffix == '.pkl':
                        df = pd.read_pickle(file_path)
                    elif file_path.suffix == '.parquet':
                        df = pd.read_parquet(file_path)
                    elif file_path.suffix == '.csv':
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    logger.info(f"Loaded {symbol} data from {file_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        if df is None:
            logger.warning(f"No data found for symbol: {symbol}")
            return pd.DataFrame()
        
        # Apply date filters if specified
        if start_date or end_date:
            df = self.filter_by_dates(df, start_date, end_date)
        
        return df
    
    def load_multiple_symbols(self, symbols: List[str],
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            df = self.load_symbol_data(symbol, start_date, end_date)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def filter_by_dates(self, df: pd.DataFrame,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: DataFrame with datetime index
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df = df.set_index('date')
            else:
                logger.warning("No datetime index or column found")
                return df
        
        # Apply filters
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def load_model_results(self, model_name: str) -> Optional[Dict]:
        """
        Load saved model results.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model results or None
        """
        results_file = self.data_dir / "results" / f"{model_name}_results.pkl"
        
        if results_file.exists():
            try:
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"Loaded results for {model_name}")
                return results
            except Exception as e:
                logger.error(f"Failed to load results for {model_name}: {e}")
        
        return None
    
    def save_model_results(self, results: Dict, model_name: str) -> None:
        """
        Save model results.
        
        Args:
            results: Dictionary with model results
            model_name: Name of the model
        """
        results_dir = self.data_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"{model_name}_results.pkl"
        
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved results for {model_name}")
        except Exception as e:
            logger.error(f"Failed to save results for {model_name}: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in data directory.
        
        Returns:
            List of symbol names
        """
        symbols = set()
        
        # Check different data directories
        for data_path in [self.data_dir, self.data_dir / "raw", self.data_dir / "processed"]:
            if data_path.exists():
                for file_path in data_path.glob("*.pkl"):
                    symbols.add(file_path.stem)
                for file_path in data_path.glob("*.parquet"):
                    symbols.add(file_path.stem)
                for file_path in data_path.glob("*.csv"):
                    symbols.add(file_path.stem)
        
        return sorted(list(symbols))
    
    def get_data_info(self, symbol: str) -> Dict:
        """
        Get information about available data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with data information
        """
        df = self.load_symbol_data(symbol)
        
        if df.empty:
            return {"symbol": symbol, "available": False}
        
        info = {
            "symbol": symbol,
            "available": True,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            },
            "missing_data": df.isnull().sum().to_dict()
        }
        
        return info
    
    def create_combined_dataset(self, symbols: List[str],
                               feature_columns: List[str],
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create a combined dataset from multiple symbols.
        
        Args:
            symbols: List of symbols to combine
            feature_columns: Columns to include
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Combined DataFrame with MultiIndex columns (symbol, feature)
        """
        data_frames = []
        
        for symbol in symbols:
            df = self.load_symbol_data(symbol, start_date, end_date)
            
            if not df.empty and all(col in df.columns for col in feature_columns):
                # Select only requested columns
                symbol_data = df[feature_columns].copy()
                
                # Create MultiIndex columns
                symbol_data.columns = pd.MultiIndex.from_product(
                    [[symbol], feature_columns]
                )
                
                data_frames.append(symbol_data)
        
        if data_frames:
            combined = pd.concat(data_frames, axis=1)
            return combined.sort_index()
        else:
            return pd.DataFrame()
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached data files.
        
        Args:
            older_than_days: Only clear files older than this many days
            
        Returns:
            Number of files deleted
        """
        deleted = 0
        cutoff_date = None
        
        if older_than_days:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cutoff_date:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time > cutoff_date:
                    continue
            
            try:
                cache_file.unlink()
                deleted += 1
                logger.info(f"Deleted cache file: {cache_file.name}")
            except Exception as e:
                logger.error(f"Failed to delete {cache_file}: {e}")
        
        return deleted


# Convenience functions
def load_data(symbol: str, data_dir: str = "data") -> pd.DataFrame:
    """Quick function to load data for a single symbol."""
    loader = DataLoader(data_dir)
    return loader.load_symbol_data(symbol)


def load_multiple_data(symbols: List[str], data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Quick function to load data for multiple symbols."""
    loader = DataLoader(data_dir)
    return loader.load_multiple_symbols(symbols)