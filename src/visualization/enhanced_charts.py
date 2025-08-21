"""
Enhanced Visualization Module with High-DPI Support
====================================================
Implements best practices for financial charts with proper styling,
high DPI support, and no font cutoffs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure high-quality visualization settings
def setup_plotting_style():
    """Configure matplotlib and seaborn for high-quality financial charts"""
    
    # Set seaborn style - Set2 palette for better contrast
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    
    # Configure matplotlib for high DPI
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Font settings to prevent cutoffs
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Layout settings to prevent cutoffs
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.05
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.05
    
    # Grid settings
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Axes settings
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    return True

# Initialize style on import
setup_plotting_style()

class FinancialChartGenerator:
    """Generate high-quality financial charts with consistent styling"""
    
    def __init__(self):
        self.crypto_color = '#FF6B6B'  # Coral red for crypto
        self.equity_color = '#4ECDC4'  # Teal for equity
        self.positive_color = '#95E77E'  # Green for positive
        self.negative_color = '#FF6B6B'  # Red for negative
        self.neutral_color = '#FFE66D'  # Yellow for neutral
        
    def create_performance_comparison(self, 
                                     crypto_returns: pd.Series,
                                     equity_returns: pd.Series,
                                     title: str = "Crypto vs Equity Performance") -> plt.Figure:
        """Create performance comparison chart with proper styling"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        crypto_cum = (1 + crypto_returns).cumprod()
        equity_cum = (1 + equity_returns).cumprod()
        
        ax1.plot(crypto_cum.index, crypto_cum.values, 
                label='Cryptocurrency', color=self.crypto_color, linewidth=2)
        ax1.plot(equity_cum.index, equity_cum.values, 
                label='Equity', color=self.equity_color, linewidth=2)
        
        ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # 2. Return Distribution
        ax2 = axes[0, 1]
        ax2.hist(crypto_returns, bins=50, alpha=0.6, label='Crypto', 
                color=self.crypto_color, edgecolor='black', linewidth=0.5)
        ax2.hist(equity_returns, bins=50, alpha=0.6, label='Equity', 
                color=self.equity_color, edgecolor='black', linewidth=0.5)
        
        ax2.set_title('Return Distribution', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Returns', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # 3. Rolling Volatility
        ax3 = axes[1, 0]
        crypto_vol = crypto_returns.rolling(30).std() * np.sqrt(252)
        equity_vol = equity_returns.rolling(30).std() * np.sqrt(252)
        
        ax3.plot(crypto_vol.index, crypto_vol.values, 
                label='Crypto', color=self.crypto_color, linewidth=2)
        ax3.plot(equity_vol.index, equity_vol.values, 
                label='Equity', color=self.equity_color, linewidth=2)
        
        ax3.set_title('30-Day Rolling Volatility (Annualized)', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Volatility', fontsize=12)
        ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Return Scatter
        ax4 = axes[1, 1]
        
        # Calculate metrics
        crypto_mean = crypto_returns.mean() * 252
        crypto_std = crypto_returns.std() * np.sqrt(252)
        equity_mean = equity_returns.mean() * 252
        equity_std = equity_returns.std() * np.sqrt(252)
        
        ax4.scatter(crypto_std, crypto_mean, s=200, color=self.crypto_color, 
                   label='Crypto', edgecolor='black', linewidth=2, alpha=0.7)
        ax4.scatter(equity_std, equity_mean, s=200, color=self.equity_color, 
                   label='Equity', edgecolor='black', linewidth=2, alpha=0.7)
        
        # Add Sharpe ratio lines
        sharpe_ratios = [0.5, 1.0, 1.5]
        x_range = np.linspace(0, max(crypto_std, equity_std) * 1.2, 100)
        for sr in sharpe_ratios:
            ax4.plot(x_range, sr * x_range, '--', alpha=0.3, 
                    label=f'Sharpe={sr}')
        
        ax4.set_title('Risk-Return Profile', fontsize=14, fontweight='bold', pad=10)
        ax4.set_xlabel('Volatility (Annual)', fontsize=12)
        ax4.set_ylabel('Return (Annual)', fontsize=12)
        ax4.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout to prevent cutoffs
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        return fig
    
    def create_portfolio_weights_chart(self, 
                                      weights: Dict[str, float],
                                      title: str = "Portfolio Allocation") -> plt.Figure:
        """Create portfolio allocation pie chart with proper styling"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Separate crypto and equity
        crypto_weights = {k: v for k, v in weights.items() if 'USD' in k}
        equity_weights = {k: v for k, v in weights.items() if 'USD' not in k}
        
        # 1. Overall allocation
        ax1.pie([sum(crypto_weights.values()), sum(equity_weights.values())],
               labels=['Cryptocurrency', 'Equity'],
               colors=[self.crypto_color, self.equity_color],
               autopct='%1.1f%%',
               startangle=90,
               explode=(0.05, 0),
               shadow=True)
        ax1.set_title('Asset Class Allocation', fontsize=14, fontweight='bold', pad=20)
        
        # 2. Individual assets
        all_labels = list(weights.keys())
        all_values = list(weights.values())
        colors = [self.crypto_color if 'USD' in label else self.equity_color 
                 for label in all_labels]
        
        # Sort by weight for better visualization
        sorted_indices = np.argsort(all_values)[::-1]
        sorted_labels = [all_labels[i] for i in sorted_indices]
        sorted_values = [all_values[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        
        # Create bar chart
        bars = ax2.barh(range(len(sorted_labels)), sorted_values, 
                       color=sorted_colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        ax2.set_yticks(range(len(sorted_labels)))
        ax2.set_yticklabels(sorted_labels)
        ax2.set_xlabel('Weight (%)', fontsize=12)
        ax2.set_title('Individual Asset Weights', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            ax2.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{value*100:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        return fig
    
    def create_walk_forward_results(self,
                                   fold_results: List[Dict],
                                   title: str = "Walk-Forward Validation Results") -> plt.Figure:
        """Create comprehensive walk-forward validation results chart"""
        
        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        folds = [r['fold'] for r in fold_results]
        returns = [r['return'] for r in fold_results]
        sharpes = [r['sharpe'] for r in fold_results]
        drawdowns = [r['max_drawdown'] for r in fold_results]
        
        # 1. Returns by Fold
        ax1 = fig.add_subplot(gs[0, :])
        colors = [self.positive_color if r > 0 else self.negative_color for r in returns]
        bars1 = ax1.bar(folds, np.array(returns) * 100, color=colors, 
                       alpha=0.7, edgecolor='black', linewidth=1)
        
        ax1.set_title('Returns by Fold', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('Return (%)', fontsize=12)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ret in zip(bars1, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret*100:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10)
        
        # 2. Sharpe Ratios
        ax2 = fig.add_subplot(gs[1, 0])
        colors = [self.positive_color if s > 0 else self.negative_color for s in sharpes]
        bars2 = ax2.bar(folds, sharpes, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=1)
        
        ax2.set_title('Sharpe Ratio by Fold', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Fold', fontsize=11)
        ax2.set_ylabel('Sharpe Ratio', fontsize=11)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # 3. Maximum Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        bars3 = ax3.bar(folds, np.array(drawdowns) * 100, color=self.negative_color, 
                       alpha=0.7, edgecolor='black', linewidth=1)
        
        ax3.set_title('Maximum Drawdown by Fold', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xlabel('Fold', fontsize=11)
        ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
        ax3.axhline(y=-20, color='orange', linestyle='--', alpha=0.5, label='Warning')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # 4. Cumulative Performance
        ax4 = fig.add_subplot(gs[1, 2])
        cumulative_returns = np.cumprod(1 + np.array(returns))
        ax4.plot(folds, cumulative_returns, 'o-', color='#2E86AB', 
                linewidth=2, markersize=8)
        ax4.fill_between(folds, 1, cumulative_returns, alpha=0.3, color='#2E86AB')
        
        ax4.set_title('Cumulative Performance', fontsize=12, fontweight='bold', pad=10)
        ax4.set_xlabel('Fold', fontsize=11)
        ax4.set_ylabel('Cumulative Return', fontsize=11)
        ax4.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance Summary Table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Calculate summary statistics
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)
        avg_dd = np.mean(drawdowns)
        best_fold = folds[np.argmax(sharpes)]
        worst_fold = folds[np.argmin(sharpes)]
        
        summary_data = [
            ['Metric', 'Value', 'Assessment'],
            ['Average Return', f'{avg_return*100:.2f}%', 'Good' if avg_return > 0.1 else 'Needs Improvement'],
            ['Average Sharpe', f'{avg_sharpe:.3f}', 'Good' if avg_sharpe > 0.8 else 'Needs Improvement'],
            ['Average Max DD', f'{avg_dd*100:.2f}%', 'Acceptable' if avg_dd > -0.25 else 'High Risk'],
            ['Best Fold', f'Fold {best_fold}', f'Sharpe: {max(sharpes):.3f}'],
            ['Worst Fold', f'Fold {worst_fold}', f'Sharpe: {min(sharpes):.3f}'],
            ['Consistency', f'{np.std(returns)*100:.2f}% std', 'Stable' if np.std(returns) < 0.15 else 'Volatile']
        ]
        
        table = ax5.table(cellText=summary_data, loc='center', cellLoc='left',
                         colWidths=[0.3, 0.3, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code assessment column
        for i in range(1, len(summary_data)):
            if 'Good' in summary_data[i][2]:
                table[(i, 2)].set_facecolor('#E8F5E9')
            elif 'Needs Improvement' in summary_data[i][2] or 'High Risk' in summary_data[i][2]:
                table[(i, 2)].set_facecolor('#FFEBEE')
            elif 'Acceptable' in summary_data[i][2] or 'Stable' in summary_data[i][2]:
                table[(i, 2)].set_facecolor('#FFF9C4')
        
        ax5.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """Save figure with high DPI and proper formatting"""
        
        # Ensure the figure uses constrained layout
        fig.set_constrained_layout(True)
        
        # Save with high DPI
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"Figure saved: {filename} (DPI: {dpi})")


class RiskDashboard:
    """Create comprehensive risk management dashboards"""
    
    def __init__(self):
        self.chart_gen = FinancialChartGenerator()
    
    def create_var_comparison_chart(self,
                                   var_results: Dict[str, Dict],
                                   title: str = "Value at Risk Comparison") -> plt.Figure:
        """Create VaR comparison chart across methods and assets"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # Prepare data
        methods = list(next(iter(var_results.values())).keys())
        assets = list(var_results.keys())
        
        # 1. Heatmap of VaR values
        ax1 = axes[0, 0]
        var_matrix = np.array([[var_results[asset][method] * 100 
                               for method in methods] for asset in assets])
        
        im = ax1.imshow(var_matrix, cmap='RdYlGn_r', aspect='auto')
        ax1.set_xticks(np.arange(len(methods)))
        ax1.set_yticks(np.arange(len(assets)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.set_yticklabels(assets)
        ax1.set_title('VaR Heatmap (95% confidence)', fontsize=14, fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('VaR (%)', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(assets)):
            for j in range(len(methods)):
                text = ax1.text(j, i, f'{var_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        # 2. Box plot by method
        ax2 = axes[0, 1]
        box_data = [[var_results[asset][method] * 100 for asset in assets] 
                   for method in methods]
        
        bp = ax2.boxplot(box_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("Set2", len(methods))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('VaR Distribution by Method', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('VaR (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Crypto vs Equity comparison
        ax3 = axes[1, 0]
        crypto_vars = [var_results[asset]['Historical'] * 100 
                      for asset in assets if 'USD' in asset]
        equity_vars = [var_results[asset]['Historical'] * 100 
                      for asset in assets if 'USD' not in asset]
        
        positions = [1, 2]
        bp1 = ax3.boxplot([crypto_vars], positions=[positions[0]], 
                         widths=0.6, patch_artist=True)
        bp2 = ax3.boxplot([equity_vars], positions=[positions[1]], 
                         widths=0.6, patch_artist=True)
        
        bp1['boxes'][0].set_facecolor(self.chart_gen.crypto_color)
        bp2['boxes'][0].set_facecolor(self.chart_gen.equity_color)
        
        ax3.set_xticks(positions)
        ax3.set_xticklabels(['Cryptocurrency', 'Equity'])
        ax3.set_title('VaR: Crypto vs Equity', fontsize=14, fontweight='bold', pad=10)
        ax3.set_ylabel('Historical VaR (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Average VaR by asset
        ax4 = axes[1, 1]
        avg_vars = {asset: np.mean(list(var_results[asset].values())) * 100 
                   for asset in assets}
        
        sorted_assets = sorted(avg_vars.keys(), key=lambda x: avg_vars[x], reverse=True)
        sorted_values = [avg_vars[asset] for asset in sorted_assets]
        colors = [self.chart_gen.crypto_color if 'USD' in asset else self.chart_gen.equity_color 
                 for asset in sorted_assets]
        
        bars = ax4.barh(range(len(sorted_assets)), sorted_values, 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        ax4.set_yticks(range(len(sorted_assets)))
        ax4.set_yticklabels(sorted_assets)
        ax4.set_xlabel('Average VaR (%)', fontsize=12)
        ax4.set_title('Average VaR by Asset', fontsize=14, fontweight='bold', pad=10)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, sorted_values):
            ax4.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        return fig


# Test the enhanced visualization
if __name__ == "__main__":
    print("Testing Enhanced Visualization Module")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    # Simulate returns
    crypto_returns = pd.Series(np.random.normal(0.002, 0.04, len(dates)), index=dates)
    equity_returns = pd.Series(np.random.normal(0.0003, 0.015, len(dates)), index=dates)
    
    # Create chart generator
    chart_gen = FinancialChartGenerator()
    
    # Test performance comparison
    fig1 = chart_gen.create_performance_comparison(crypto_returns, equity_returns)
    chart_gen.save_figure(fig1, 'test_performance_comparison.png')
    
    # Test portfolio weights
    weights = {
        'BTCUSD': 0.05,
        'ETHUSD': 0.05,
        'SOLUSD': 0.03,
        'BNBUSD': 0.02,
        'SPY': 0.35,
        'QQQ': 0.25,
        'DIA': 0.15,
        'IWM': 0.10
    }
    
    fig2 = chart_gen.create_portfolio_weights_chart(weights)
    chart_gen.save_figure(fig2, 'test_portfolio_weights.png')
    
    # Test walk-forward results
    fold_results = [
        {'fold': 1, 'return': 0.12, 'sharpe': 0.85, 'max_drawdown': -0.15},
        {'fold': 2, 'return': 0.08, 'sharpe': 0.72, 'max_drawdown': -0.18},
        {'fold': 3, 'return': 0.15, 'sharpe': 0.95, 'max_drawdown': -0.12},
        {'fold': 4, 'return': -0.05, 'sharpe': -0.25, 'max_drawdown': -0.25},
        {'fold': 5, 'return': 0.10, 'sharpe': 0.68, 'max_drawdown': -0.20}
    ]
    
    fig3 = chart_gen.create_walk_forward_results(fold_results)
    chart_gen.save_figure(fig3, 'test_walk_forward_results.png')
    
    print("\nVisualization tests complete!")
    print("Check generated PNG files for quality.")