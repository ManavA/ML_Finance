"""
"""

import json
import yaml
from pathlib import Path
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {path.suffix}")

def get_default_config() -> Dict[str, Any]:
    return {
        'data': {
            'sources': {
                'priority': ['polygon', 'cmc', 's3'],
                'polygon': {
                    'api_key': os.getenv('POLYGON_API_KEY', '')
                },
                'cmc': {
                    'api_key': os.getenv('CMC_API_KEY', '')
                },
                's3': {
                    'bucket': 'crypto-ml-data',
                    'region': 'us-east-1'
                }
            },
            'cache': {
                'enabled': True,
                'directory': 'data/cache',
                'ttl': 3600
            }
        },
        'models': {
            'default': 'xgboost',
            'hyperparameters': {
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.01
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.01
                },
                'catboost': {
                    'iterations': 100,
                    'depth': 5,
                    'learning_rate': 0.01
                }
            }
        },
        'trading': {
            'initial_capital': 10000,
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'commission': 0.001
        },
        'backtesting': {
            'walk_forward': {
                'enabled': True,
                'train_periods': 252,
                'test_periods': 63,
                'step_periods': 21
            }
        },
        'logging': {
            'level': 'INFO',
            'file': 'trading.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

def save_config(config: Dict[str, Any], path: str):
    path = Path(path)
    
    if path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    elif path.suffix in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration format: {path.suffix}")

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result