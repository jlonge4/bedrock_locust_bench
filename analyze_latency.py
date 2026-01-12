#!/usr/bin/env python3
"""
Regression Analysis for Bedrock Locust Benchmark Results
Analyzes the statistical significance of latency differences across service tiers.

Usage:
    python analyze_latency.py --results-dir test_results
    python analyze_latency.py --consolidated-csv test_results/consolidated_results_20250112.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import glob


def load_data(results_dir=None, consolidated_csv=None):
    """Load data from either results directory or consolidated CSV."""
    if consolidated_csv:
        print(f"Loading consolidated CSV: {consolidated_csv}")
        df = pd.read_csv(consolidated_csv)
    elif results_dir:
        # Find the most recent consolidated results file
        results_path = Path(results_dir)
        csv_files = list(results_path.glob("consolidated_results_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No consolidated_results_*.csv found in {results_dir}")
        
        # Get most recent file
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading most recent consolidated CSV: {latest_csv}")
        df = pd.read_csv(latest_csv)
    else:
        raise ValueError("Must provide either --results-dir or --consolidated-csv")
    
    return df


def prepare_features(df):
    """Prepare features for regression analysis."""
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Extract configuration components if not already separated
    if 'service_tier' not in data.columns and 'Test Configuration' in data.columns:
        # Parse test configuration: "us_priority_medium_30users"
        config_parts = data['Test Configuration'].str.split('_', expand=True)
        data['region_prefix'] = config_parts[0]
        data['service_tier'] = config_parts[1]
        data['prompt_size'] = config_parts[2]
        data['user_count'] = config_parts[3].str.replace('users', '').astype(int)
    
    # One-hot encode service tier (use 'default' as baseline)
    tier_dummies = pd.get_dummies(data['service_tier'], prefix='tier', drop_first=False)
    if 'tier_default' in tier_dummies.columns:
        tier_dummies = tier_dummies.drop('tier_default', axis=1)
    
    # One-hot encode prompt size (use 'small' as baseline)
    size_dummies = pd.get_dummies(data['prompt_size'], prefix='size', drop_first=False)
    if 'size_small' in size_dummies.columns:
        size_dummies = size_dummies.drop('size_small', axis=1)
    
    # Combine features
    feature_cols = ['user_count']
    X = pd.concat([
        data[feature_cols],
        tier_dummies,
        size_dummies
    ], axis=1)
    
    return X, data


def run_regression_analysis(df, target_metric='Average Response Time'):
    """Run regression analysis for a specific latency metric."""
    X, data = prepare_features(df)
    
    # Get target variable
    if target_metric not in data.columns:
        print(f"Warning: {target_metric} not found in data. Available columns:")
        print(data.columns.tolist())
        return None
    
    y = data[target_metric]
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()
    
    return model, X_with_const, data


def print_regression_summary(model, metric_name):
    """Print formatted regression results."""
    print(f"\n{'='*80}")
    print(f"REGRESSION ANALYSIS: {metric_name}")
    print('='*80)
    print(model.summary())
    
    print(f"\n{metric_name} - Service Tier Impact (vs default baseline):")
    print("-" * 70)
    
    for param_name in model.params.index:
        if param_name.startswith('tier_'):
            tier = param_name.replace('tier_', '')
            beta = model.params[param_name]
            pval = model.pvalues[param_name]
            ci_lower, ci_upper = model.conf_int().loc[param_name]
            
            # Significance stars
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            
            # Interpretation
            direction = "faster" if beta < 0 else "slower"
            
            print(f"  {tier.upper():10s}: β = {beta:+8.2f} ms ({direction} than default)")
            print(f"              p-value = {pval:.4f} {sig}")
            print(f"              95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
            print()


def check_multicollinearity(X):
    """Check for multicollinearity using VIF."""
    print("\n" + "="*80)
    print("MULTICOLLINEARITY CHECK (Variance Inflation Factor)")
    print("="*80)
    print("VIF > 10 indicates high multicollinearity\n")
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print(vif_data.to_string(index=False))
    print()


def create_comparison_plots(df, output_dir):
    """Create visualization plots for the regression analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    X, data = prepare_features(df)
    
    # 1. Box plots by service tier
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['50%', '95%', 'Average Response Time']
    
    for idx, metric in enumerate(metrics):
        if metric in data.columns:
            sns.boxplot(data=data, x='service_tier', y=metric, ax=axes[idx])
            axes[idx].set_title(f'{metric} by Service Tier')
            axes[idx].set_xlabel('Service Tier')
            axes[idx].set_ylabel('Latency (ms)')
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'service_tier_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'service_tier_boxplots.png'}")
    plt.close()
    
    # 2. Service tier comparison by prompt size
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, metric in enumerate(metrics):
        if metric in data.columns:
            pivot_data = data.pivot_table(
                values=metric,
                index='prompt_size',
                columns='service_tier',
                aggfunc='mean'
            )
            pivot_data.plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{metric} by Prompt Size & Service Tier')
            axes[idx].set_xlabel('Prompt Size')
            axes[idx].set_ylabel('Latency (ms)')
            axes[idx].legend(title='Service Tier')
            axes[idx].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'tier_by_prompt_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'tier_by_prompt_size.png'}")
    plt.close()
    
    # 3. Scatter plot with regression line
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, metric in enumerate(metrics):
        if metric in data.columns:
            for tier in data['service_tier'].unique():
                tier_data = data[data['service_tier'] == tier]
                axes[idx].scatter(tier_data['user_count'], tier_data[metric], 
                                alpha=0.6, label=tier)
            
            axes[idx].set_title(f'{metric} vs User Count')
            axes[idx].set_xlabel('Concurrent Users')
            axes[idx].set_ylabel('Latency (ms)')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'latency_vs_users.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'latency_vs_users.png'}")
    plt.close()


def print_summary_stats(df):
    """Print summary statistics by service tier."""
    X, data = prepare_features(df)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY SERVICE TIER")
    print("="*80)
    
    metrics = ['50%', '95%', 'Average Response Time']
    
    for metric in metrics:
        if metric in data.columns:
            print(f"\n{metric}:")
            print(data.groupby('service_tier')[metric].describe().round(2))


def main():
    parser = argparse.ArgumentParser(
        description='Regression analysis for Bedrock Locust benchmark results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory containing test results (will use most recent consolidated CSV)'
    )
    parser.add_argument(
        '--consolidated-csv',
        type=str,
        help='Path to specific consolidated results CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Directory to save analysis plots and outputs'
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(results_dir=args.results_dir, consolidated_csv=args.consolidated_csv)
    
    print(f"\nLoaded {len(df)} test results")
    print(f"Columns: {df.columns.tolist()}")
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Run regression for each latency metric
    metrics = {
        '50%': 'P50 Latency',
        '95%': 'P95 Latency',
        'Average Response Time': 'Mean Latency'
    }
    
    for col_name, display_name in metrics.items():
        if col_name in df.columns:
            model, X, data = run_regression_analysis(df, target_metric=col_name)
            if model:
                print_regression_summary(model, display_name)
    
    # Check multicollinearity on one model
    model, X, data = run_regression_analysis(df, target_metric='95%')
    if model:
        check_multicollinearity(X.drop('const', axis=1))
    
    # Create plots
    create_comparison_plots(df, args.output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"Plots saved to: {args.output_dir}/")
    print('='*80)


if __name__ == '__main__':
    main()