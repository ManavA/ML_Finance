"""
Setup script for ML Finance Trading System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename="requirements.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ml-finance-trading",
    version="1.0.0",
    author="ML Finance Team",
    author_email="team@mlfinance.com",
    description="A comprehensive machine learning system for cryptocurrency and stock trading",
    long_description=read_long_description() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-finance-trading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies only
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "click>=8.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "ml": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
        ],
        "deep": [
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
        ],
        "rl": [
            "stable-baselines3>=1.5.0",
            "gymnasium>=0.26.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.3.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
        ],
        "web": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
        "all": [
            # Include everything
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
            "torch>=1.10.0",
            "stable-baselines3>=1.5.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.3.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-finance=cli.cli:cli",
            "mlf=cli.cli:cli",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)