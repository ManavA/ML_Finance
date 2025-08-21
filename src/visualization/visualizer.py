# src/analysis/visualizer.py
"""
Visualization tools for strategy analysis.
Clean, publication-ready plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StrategyVisualizer:
    """Visualization tools for strategy analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_equity_curves(self, 
                          results_list: List,
                          title: str = "Strategy Equity Curves") -> plt.Figure:
        """
        Plot equity curves for multiple strategies.
        
        Args:
            results_list: List of BacktestResults
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for result in results_list:
            ax.plot(result.equity_curve.index, 
                   result.equity_curve.values,
                   label=result.strategy_name,
                   linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(self,
                                 results_list: List,
                                 title: str = "Returns Distribution") -> plt.Figure:
        """
        Plot returns distribution for multiple strategies.
        
        Args:
            results_list: List of BacktestResults
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1 = axes[0]
        for result in results_list:
            ax1.hist(result.returns.dropna() * 100, 
                    bins=50, 
                    alpha=0.5, 
                    label=result.strategy_name,
                    edgecolor='black')
        
        ax1.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Daily Returns (%)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        returns_data = []
        labels = []
        for result in results_list:
            returns_data.append(result.returns.dropna() * 100)
            labels.append(result.strategy_name)
        
        bp = ax2.boxplot(returns_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(results_list))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Returns Box Plot', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Returns (%)', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self,
                      results_list: List,
                      title: str = "Strategy Drawdowns") -> plt.Figure:
        """
        Plot drawdown for multiple strategies.
        
        Args:
            results_list: List of BacktestResults
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for result in results_list:
            ax.fill_between(result.drawdown.index,
                          result.drawdown.values * 100,
                          0,
                          alpha=0.3,
                          label=result.strategy_name)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_return(self,
                        results_list: List,
                        title: str = "Risk-Return Profile") -> plt.Figure:
        """
        Plot risk-return scatter plot.
        
        Args:
            results_list: List of BacktestResults
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract metrics
        returns = []
        volatilities = []
        sharpes = []
        names = []
        
        for result in results_list:
            returns.append(result.metrics['annual_return'] * 100)
            volatilities.append(result.metrics['volatility'] * 100)
            sharpes.append(result.metrics['sharpe_ratio'])
            names.append(result.strategy_name)
        
        # Create scatter plot
        scatter = ax.scatter(volatilities, returns, 
                           s=200,
                           c=sharpes,
                           cmap='RdYlGn',
                           edgecolors='black',
                           linewidth=2,
                           alpha=0.7)
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, 
                       (volatilities[i], returns[i]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=10)
        
        # Add efficient frontier line (simplified)
        ax.plot([0, max(volatilities)], 
               [0, max(volatilities) * max(sharpes)],
               'k--', alpha=0.3, label='Simplified Efficient Frontier')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add zero lines
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self,
                              results_list: List,
                              title: str = "Strategy Returns Correlation") -> plt.Figure:
        """
        Plot correlation matrix of strategy returns.
        
        Args:
            results_list: List of BacktestResults
            title: Plot title
            
        Returns:
            Figure object
        """
        # Create returns DataFrame
        returns_df = pd.DataFrame()
        for result in results_list:
            returns_df[result.strategy_name] = result.returns
        
        # Calculate correlation
        corr = returns_df.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, 
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns_heatmap(self,
                                    result,
                                    title: Optional[str] = None) -> plt.Figure:
        """
        Plot monthly returns heatmap for a single strategy.
        
        Args:
            result: BacktestResults object
            title: Plot title
            
        Returns:
            Figure object
        """
        # Calculate monthly returns
        monthly_returns = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Pivot by year and month
        monthly_returns_df = pd.DataFrame(monthly_returns, columns=['returns'])
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        # Create pivot table
        pivot_table = monthly_returns_df.pivot(index='year', 
                                               columns='month', 
                                               values='returns')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_table * 100,
                   annot=True,
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=0,
                   cbar_kws={'label': 'Monthly Return (%)'},
                   ax=ax)
        
        # Set labels
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        if title is None:
            title = f"Monthly Returns - {result.strategy_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Format month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self,
                            results_list: List,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive summary report.
        
        Args:
            results_list: List of BacktestResults
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 20))
        
        # Create subplots
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
        
        # 1. Equity curves
        ax1 = fig.add_subplot(gs[0, :])
        for result in results_list:
            ax1.plot(result.equity_curve.index, 
                    result.equity_curve.values,
                    label=result.strategy_name,
                    linewidth=2)
        ax1.set_title('Equity Curves', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        for result in results_list:
            ax2.fill_between(result.drawdown.index,
                           result.drawdown.values * 100,
                           0,
                           alpha=0.3,
                           label=result.strategy_name)
        ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk-Return
        ax3 = fig.add_subplot(gs[2, 0])
        returns = [r.metrics['annual_return'] * 100 for r in results_list]
        volatilities = [r.metrics['volatility'] * 100 for r in results_list]
        sharpes = [r.metrics['sharpe_ratio'] for r in results_list]
        names = [r.strategy_name for r in results_list]
        
        scatter = ax3.scatter(volatilities, returns, 
                            s=150,
                            c=sharpes,
                            cmap='RdYlGn',
                            edgecolors='black',
                            linewidth=1.5,
                            alpha=0.7)
        
        for i, name in enumerate(names):
            ax3.annotate(name, 
                        (volatilities[i], returns[i]),
                        xytext=(3, 3),
                        textcoords='offset points',
                        fontsize=8)
        
        ax3.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Annual Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Returns distribution
        ax4 = fig.add_subplot(gs[2, 1])
        for result in results_list:
            ax4.hist(result.returns.dropna() * 100, 
                    bins=30, 
                    alpha=0.5, 
                    label=result.strategy_name,
                    density=True)
        ax4.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Daily Returns (%)')
        ax4.set_ylabel('Density')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # 5. Metrics table
        ax5 = fig.add_subplot(gs[3:, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create metrics table
        metrics_data = []
        for result in results_list:
            row = [
                result.strategy_name,
                f"{result.metrics['total_return']*100:.1f}%",
                f"{result.metrics['annual_return']*100:.1f}%",
                f"{result.metrics['volatility']*100:.1f}%",
                f"{result.metrics['sharpe_ratio']:.2f}",
                f"{result.metrics['sortino_ratio']:.2f}",
                f"{result.metrics['max_drawdown']*100:.1f}%",
                f"{result.metrics['win_rate']*100:.1f}%",
                f"{result.metrics['profit_factor']:.2f}"
            ]
            metrics_data.append(row)
        
        headers = ['Strategy', 'Total Return', 'Annual Return', 'Volatility', 
                  'Sharpe', 'Sortino', 'Max DD', 'Win Rate', 'Profit Factor']
        
        table = ax5.table(cellText=metrics_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.15] + [0.106] * 8)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(metrics_data) + 1):
            for j in range(len(headers)):
                if j == 0:  # Strategy name column
                    table[(i, j)].set_facecolor('#f0f0f0')
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.suptitle('Strategy Performance Summary Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig