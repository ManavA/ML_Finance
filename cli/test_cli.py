#!/usr/bin/env python3
"""
Test script for the Crypto ML Trading CLI
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a CLI command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_cli():
    """Test basic CLI functionality"""
    print("Testing Crypto ML Trading CLI...")
    print("=" * 50)
    
    # Change to CLI directory
    cli_dir = Path(__file__).parent
    original_dir = Path.cwd()
    
    try:
        # Test basic CLI help
        print("\n1. Testing CLI help...")
        success, stdout, stderr = run_command("python cli.py --help")
        if success:
            print("✓ CLI help works")
        else:
            print(f"✗ CLI help failed: {stderr}")
        
        # Test version command
        print("\n2. Testing version command...")
        success, stdout, stderr = run_command("python cli.py version")
        if success:
            print("✓ Version command works")
        else:
            print(f"✗ Version command failed: {stderr}")
        
        # Test config command
        print("\n3. Testing config command...")
        success, stdout, stderr = run_command("python cli.py config")
        if success:
            print("✓ Config command works")
        else:
            print(f"✗ Config command failed: {stderr}")
        
        # Test data group
        print("\n4. Testing data commands...")
        success, stdout, stderr = run_command("python cli.py data --help")
        if success:
            print("✓ Data group works")
        else:
            print(f"✗ Data group failed: {stderr}")
        
        # Test train group
        print("\n5. Testing train commands...")
        success, stdout, stderr = run_command("python cli.py train --help")
        if success:
            print("✓ Train group works")
        else:
            print(f"✗ Train group failed: {stderr}")
        
        # Test backtest group
        print("\n6. Testing backtest commands...")
        success, stdout, stderr = run_command("python cli.py backtest --help")
        if success:
            print("✓ Backtest group works")
        else:
            print(f"✗ Backtest group failed: {stderr}")
        
        # Test analyze group
        print("\n7. Testing analyze commands...")
        success, stdout, stderr = run_command("python cli.py analyze --help")
        if success:
            print("✓ Analyze group works")
        else:
            print(f"✗ Analyze group failed: {stderr}")
        
        # Test trade group
        print("\n8. Testing trade commands...")
        success, stdout, stderr = run_command("python cli.py trade --help")
        if success:
            print("✓ Trade group works")
        else:
            print(f"✗ Trade group failed: {stderr}")
        
        print("\n" + "=" * 50)
        print("CLI basic functionality test complete!")
        
        # Test imports
        print("\n9. Testing imports...")
        try:
            from cli.commands import data, backtest, train, analyze, trade
            from cli.core.config import get_default_config
            from cli.utils.logger import setup_logger
            print("✓ All imports successful")
        except Exception as e:
            print(f"✗ Import failed: {e}")
        
        print("\nTo run full functionality tests:")
        print("1. Ensure data sources are configured in .env")
        print("2. Run: python cli.py data fetch --symbols BTC --start 2024-01-01 --end 2024-01-02")
        print("3. Run: python cli.py train model --symbol BTC --model-type xgboost")
        print("4. Run: python cli.py backtest run --symbol BTC --strategy ma_crossover")
        
    finally:
        pass

if __name__ == "__main__":
    test_cli()