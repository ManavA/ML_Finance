#!/usr/bin/env python3
"""
Visualization module for ML comparison results
Creates comprehensive charts and reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ResultsVisualizer:
    """
    Create visualizations for ML comparison results
    """
    
    def __init__(self, results_dir: str = 'results/ml_comparison'):
        """
        Initialize visualizer
        
        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self) -> Tuple[Dict, List]:
        """Load saved results"""
        # Load comparison metrics
        metrics_file = self.results_dir / 'comparison_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        # Load detailed results
        results_file = self.results_dir / 'detailed_results.pkl'
        if results_file.exists():
            with open(results_file, 'rb') as f:
                detailed = pickle.load(f)
        else:
            detailed = []
        
        return metrics, detailed
    
    def plot_ml_vs_traditional_comparison(self, metrics: Dict):
        """
        Create bar chart comparing ML vs traditional strategies
        """
        if not metrics:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML vs Traditional Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, (market, market_metrics) in enumerate(metrics.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Prepare data
            categories = ['ML Models', 'Traditional']
            sharpe_values = [market_metrics['ml_sharpe'], market_metrics['trad_sharpe']]
            return_values = [market_metrics['ml_return'], market_metrics['trad_return']]
            
            # Create grouped bar chart
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, sharpe_values, width, label='Sharpe Ratio', color='steelblue')
            bars2 = ax.bar(x + width/2, [r/10 for r in return_values], width, label='Return (÷10)', color='coral')
            
            ax.set_xlabel('Strategy Type')
            ax.set_ylabel('Performance')
            ax.set_title(f'{market.upper()} Market')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height*10:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ml_vs_traditional_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hypothesis_test_results(self, metrics: Dict):
        """
        Visualize hypothesis test results
        """
        if 'crypto' not in metrics or 'equity' not in metrics:
            print("Need both crypto and equity results for hypothesis test")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ML Advantage comparison
        markets = ['Crypto', 'Equity']
        sharpe_advantages = [
            metrics['crypto']['sharpe_advantage'],
            metrics['equity']['sharpe_advantage']
        ]
        return_advantages = [
            metrics['crypto']['return_advantage'],
            metrics['equity']['return_advantage']
        ]
        
        # Sharpe advantage
        colors = ['green' if x > 0 else 'red' for x in sharpe_advantages]
        bars1 = ax1.bar(markets, sharpe_advantages, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Sharpe Ratio Advantage')
        ax1.set_title('ML Advantage by Market (Sharpe Ratio)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, sharpe_advantages):
            ax1.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.3f}', ha='center', 
                    va='bottom' if val > 0 else 'top')
        
        # Return advantage
        colors = ['green' if x > 0 else 'red' for x in return_advantages]
        bars2 = ax2.bar(markets, return_advantages, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Return Advantage (%)')
        ax2.set_title('ML Advantage by Market (Returns)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, return_advantages):
            ax2.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.1f}%', ha='center',
                    va='bottom' if val > 0 else 'top')
        
        # Add hypothesis test result
        crypto_adv = metrics['crypto']['sharpe_advantage']
        equity_adv = metrics['equity']['sharpe_advantage']
        difference = crypto_adv - equity_adv
        
        fig.suptitle(f'Hypothesis Test: Crypto ML Advantage - Equity ML Advantage = {difference:.3f}',
                    fontsize=14, fontweight='bold',
                    color='green' if difference > 0 else 'red')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'hypothesis_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_heatmap(self, detailed_results: List):
        """
        Create heatmap of model performance across assets
        """
        if not detailed_results:
            print("No detailed results to plot")
            return
        
        # Prepare data matrix
        model_names = set()
        symbol_names = set()
        performance_data = {}
        
        for result in detailed_results:
            symbol = result['symbol']
            symbol_names.add(symbol)
            
            if isinstance(result['results'], pd.DataFrame):
                for idx in result['results'].index:
                    if hasattr(idx, '__len__') and len(idx) >= 2:
                        model = idx[1] if isinstance(idx, tuple) else idx
                    else:
                        model = str(idx)
                    
                    model_names.add(model)
                    
                    # Get Sharpe ratio
                    try:
                        sharpe = result['results'].loc[idx][('sharpe_ratio', 'mean')]
                    except:
                        sharpe = 0
                    
                    performance_data[(model, symbol)] = sharpe
        
        # Create matrix
        models = sorted(list(model_names))
        symbols = sorted(list(symbol_names))
        
        matrix = np.zeros((len(models), len(symbols)))
        for i, model in enumerate(models):
            for j, symbol in enumerate(symbols):
                matrix[i, j] = performance_data.get((model, symbol), 0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, 
                   xticklabels=symbols,
                   yticklabels=models,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0,
                   cbar_kws={'label': 'Sharpe Ratio'})
        
        plt.title('Model Performance Heatmap (Sharpe Ratio)', fontsize=14, fontweight='bold')
        plt.xlabel('Asset')
        plt.ylabel('Model/Strategy')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, metrics: Dict, detailed_results: List):
        """
        Create interactive Plotly dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ML vs Traditional Performance',
                          'Hypothesis Test Results',
                          'Model Rankings',
                          'Market Efficiency Analysis'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'box'}]]
        )
        
        # 1. ML vs Traditional Performance
        if metrics:
            for market, market_metrics in metrics.items():
                fig.add_trace(
                    go.Bar(
                        name=f'{market} ML',
                        x=['Sharpe', 'Return/10'],
                        y=[market_metrics['ml_sharpe'], market_metrics['ml_return']/10],
                        marker_color='steelblue'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(
                        name=f'{market} Trad',
                        x=['Sharpe', 'Return/10'],
                        y=[market_metrics['trad_sharpe'], market_metrics['trad_return']/10],
                        marker_color='coral'
                    ),
                    row=1, col=1
                )
        
        # 2. Hypothesis Test
        if 'crypto' in metrics and 'equity' in metrics:
            advantages = [
                metrics['crypto']['sharpe_advantage'],
                metrics['equity']['sharpe_advantage']
            ]
            fig.add_trace(
                go.Bar(
                    x=['Crypto', 'Equity'],
                    y=advantages,
                    marker_color=['green' if x > 0 else 'red' for x in advantages],
                    text=[f'{x:.3f}' for x in advantages],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. Model Rankings (scatter plot)
        if detailed_results:
            model_scores = {}
            for result in detailed_results:
                if isinstance(result['results'], pd.DataFrame):
                    for idx in result['results'].index:
                        if hasattr(idx, '__len__') and len(idx) >= 2:
                            model = idx[1] if isinstance(idx, tuple) else idx
                        else:
                            model = str(idx)
                        
                        try:
                            sharpe = result['results'].loc[idx][('sharpe_ratio', 'mean')]
                            returns = result['results'].loc[idx][('total_return', 'mean')]
                            
                            if model not in model_scores:
                                model_scores[model] = {'sharpe': [], 'returns': []}
                            
                            model_scores[model]['sharpe'].append(sharpe)
                            model_scores[model]['returns'].append(returns)
                        except:
                            pass
            
            for model, scores in model_scores.items():
                if scores['sharpe'] and scores['returns']:
                    fig.add_trace(
                        go.Scatter(
                            x=[np.mean(scores['returns'])],
                            y=[np.mean(scores['sharpe'])],
                            mode='markers+text',
                            name=model,
                            text=[model],
                            textposition='top center',
                            marker=dict(size=10)
                        ),
                        row=2, col=1
                    )
        
        # 4. Market Efficiency Box Plot
        if detailed_results:
            for result in detailed_results:
                market = result['market']
                if isinstance(result['results'], pd.DataFrame):
                    sharpe_values = []
                    for idx in result['results'].index:
                        try:
                            sharpe = result['results'].loc[idx][('sharpe_ratio', 'mean')]
                            sharpe_values.append(sharpe)
                        except:
                            pass
                    
                    if sharpe_values:
                        fig.add_trace(
                            go.Box(
                                y=sharpe_values,
                                name=f"{result['symbol']}",
                                boxmean='sd'
                            ),
                            row=2, col=2
                        )
        
        # Update layout
        fig.update_layout(
            title_text="ML Trading Strategy Analysis Dashboard",
            showlegend=True,
            height=800,
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_xaxes(title_text="Market", row=1, col=2)
        fig.update_xaxes(title_text="Mean Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Asset", row=2, col=2)
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="ML Advantage", row=1, col=2)
        fig.update_yaxes(title_text="Mean Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        
        # Save as HTML
        fig.write_html(self.figures_dir / 'interactive_dashboard.html')
        fig.show()
        
        print(f"Interactive dashboard saved to {self.figures_dir / 'interactive_dashboard.html'}")
    
    def generate_latex_tables(self, metrics: Dict, detailed_results: List):
        """
        Generate LaTeX tables for academic paper
        """
        latex_dir = self.results_dir / 'latex'
        latex_dir.mkdir(exist_ok=True)
        
        # Main results table
        if metrics:
            with open(latex_dir / 'main_results.tex', 'w') as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{ML vs Traditional Strategy Performance}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Market & ML Sharpe & Trad Sharpe & ML Return (\\%) & Trad Return (\\%) \\\\\n")
                f.write("\\hline\n")
                
                for market, m in metrics.items():
                    f.write(f"{market.capitalize()} & {m['ml_sharpe']:.3f} & {m['trad_sharpe']:.3f} & ")
                    f.write(f"{m['ml_return']:.2f} & {m['trad_return']:.2f} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
        
        # Hypothesis test results
        if 'crypto' in metrics and 'equity' in metrics:
            with open(latex_dir / 'hypothesis_test.tex', 'w') as f:
                crypto_adv = metrics['crypto']['sharpe_advantage']
                equity_adv = metrics['equity']['sharpe_advantage']
                diff = crypto_adv - equity_adv
                
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Hypothesis Test: ML Advantage in Crypto vs Equity Markets}\n")
                f.write("\\begin{tabular}{lc}\n")
                f.write("\\hline\n")
                f.write("Metric & Value \\\\\n")
                f.write("\\hline\n")
                f.write(f"Crypto ML Advantage (Sharpe) & {crypto_adv:.3f} \\\\\n")
                f.write(f"Equity ML Advantage (Sharpe) & {equity_adv:.3f} \\\\\n")
                f.write(f"Difference & {diff:.3f} \\\\\n")
                f.write(f"Hypothesis & {'Supported' if diff > 0 else 'Not Supported'} \\\\\n")
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
        
        print(f"LaTeX tables saved to {latex_dir}")

def main():
    """Generate all visualizations"""
    visualizer = ResultsVisualizer()
    
    # Load results
    metrics, detailed = visualizer.load_results()
    
    if not metrics and not detailed:
        print("No results found. Run ML comparison first.")
        return
    
    # Generate all visualizations
    print("Generating visualizations...")
    
    if metrics:
        visualizer.plot_ml_vs_traditional_comparison(metrics)
        visualizer.plot_hypothesis_test_results(metrics)
    
    if detailed:
        visualizer.plot_model_performance_heatmap(detailed)
    
    if metrics or detailed:
        visualizer.create_interactive_dashboard(metrics, detailed)
        visualizer.generate_latex_tables(metrics, detailed)
    
    print(f"\nAll visualizations saved to {visualizer.figures_dir}")

if __name__ == "__main__":
    main()