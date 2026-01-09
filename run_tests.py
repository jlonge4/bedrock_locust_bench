#!/usr/bin/env python3
"""
Orchestration script to run Locust tests across multiple permutations
and generate comparison charts for latency metrics.
"""

import argparse
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
from itertools import product


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Bedrock Converse API load tests with Locust across multiple configurations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (us region, all tiers, all prompts, 30/60/90 users)
  python run_tests.py

  # Specify custom model and results directory
  python run_tests.py --model-id amazon.nova-2-lite-v1:0 --results-dir test_results/nova_run

  # Test only specific configurations
  python run_tests.py --region-prefixes us --service-tiers priority,flex --user-counts 30,60

  # Include failures in latency metrics
  python run_tests.py --include-failures

  # Quick test with shorter duration
  python run_tests.py --test-duration 30s --user-counts 10
        """
    )
    
    parser.add_argument(
        '--model-id', '--base-model-id',
        dest='model_id',
        default=os.getenv('BASE_MODEL_ID', 'amazon.nova-premier-v1:0'),
        help='Base Bedrock model ID without region prefix (default: amazon.nova-premier-v1:0)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path(os.getenv('RESULTS_DIR', 'test_results')),
        help='Directory to store test results (default: test_results)'
    )
    
    parser.add_argument(
        '--region-prefixes',
        default=os.getenv('REGION_PREFIXES', 'us'),
        help='Comma-separated region prefixes to test (default: us). Options: us, global'
    )
    
    parser.add_argument(
        '--service-tiers',
        default='default,priority,flex',
        help='Comma-separated service tiers to test (default: default,priority,flex)'
    )
    
    parser.add_argument(
        '--prompt-sizes',
        default='small,medium,large',
        help='Comma-separated prompt sizes to test (default: small,medium,large)'
    )
    
    parser.add_argument(
        '--user-counts',
        default=os.getenv('USER_COUNTS', '30,60,90'),
        help='Comma-separated user counts to test (default: 30,60,90)'
    )
    
    parser.add_argument(
        '--test-duration',
        default='1m',
        help='Duration for each test run (default: 1m). Examples: 30s, 2m, 5m'
    )
    
    parser.add_argument(
        '--spawn-rate',
        type=int,
        default=10,
        help='Users spawned per second (default: 10)'
    )
    
    parser.add_argument(
        '--include-failures',
        action='store_true',
        default=os.getenv('INCLUDE_FAILURES', 'false').lower() == 'true',
        help='Include failed requests in latency metrics (default: false)'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt and run tests immediately'
    )
    
    return parser.parse_args()


def setup_directories(results_dir):
    """Create necessary directories for results."""
    results_dir.mkdir(exist_ok=True)
    (results_dir / 'charts').mkdir(exist_ok=True)


def run_locust_test(model_id, region_prefix, service_tier, prompt_size, users, spawn_rate, test_id, results_dir, test_duration):
    """
    Run a single Locust test with the specified configuration.
    
    Returns:
        dict: Test results including metrics
    """
    print(f"\n{'='*80}")
    print(f"Running test {test_id}:")
    print(f"  Model: {region_prefix}.{model_id}")
    print(f"  Service Tier: {service_tier}")
    print(f"  Prompt Size: {prompt_size}")
    print(f"  Users: {users}")
    print(f"{'='*80}\n")
    
    # Prepare environment variables
    env = os.environ.copy()
    env['BASE_MODEL_ID'] = model_id
    env['REGION_PREFIX'] = region_prefix
    env['SERVICE_TIER'] = service_tier
    env['PROMPT_SIZE'] = prompt_size
    env['RESULTS_DIR'] = str(results_dir)  # Pass results dir to locustfile
    
    # Output files
    csv_file = results_dir / f"test_{test_id}_stats.csv"
    html_file = results_dir / f"test_{test_id}_report.html"
    
    # Construct Locust command
    cmd = [
        'locust',
        '-f', 'locustfile.py',
        '--headless',
        '--users', str(users),
        '--spawn-rate', str(spawn_rate),
        '--run-time', test_duration,
        '--csv', str(results_dir / f"test_{test_id}"),
        '--html', str(html_file),
        '--only-summary',
    ]
    
    try:
        # Run Locust
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # Parse results from CSV
        stats_file = results_dir / f"test_{test_id}_stats.csv"
        if stats_file.exists():
            df = pd.read_csv(stats_file)
            # Filter for the actual request type (not aggregated stats)
            df = df[df['Type'] == 'Converse']
            
            if not df.empty:
                metrics = {
                    'test_id': test_id,
                    'region_prefix': region_prefix,
                    'service_tier': service_tier,
                    'prompt_size': prompt_size,
                    'users': users,
                    'total_requests': df['Request Count'].iloc[0],
                    'failure_count': df['Failure Count'].iloc[0],
                    'avg_response_time': df['Average Response Time'].iloc[0],
                    'min_response_time': df['Min Response Time'].iloc[0],
                    'max_response_time': df['Max Response Time'].iloc[0],
                    'p50': df['50%'].iloc[0] if '50%' in df.columns else df['Median Response Time'].iloc[0],
                    'p95': df['95%'].iloc[0] if '95%' in df.columns else None,
                    'p99': df['99%'].iloc[0] if '99%' in df.columns else None,
                    'requests_per_sec': df['Requests/s'].iloc[0],
                }
                
                # Read token statistics from JSONL file for this specific test (only successful requests)
                token_file = results_dir / 'token_data.jsonl'
                if token_file.exists():
                    try:
                        input_tokens = []
                        output_tokens = []
                        response_times = []
                        
                        # Read only tokens from successful requests in the current test configuration
                        with open(token_file, 'r') as f:
                            for line in f:
                                token_record = json.loads(line.strip())
                                # Match this test's configuration and only include successful requests
                                if (token_record.get('region_prefix') == region_prefix and
                                    token_record.get('service_tier') == service_tier and
                                    token_record.get('prompt_size') == prompt_size and
                                    token_record.get('success', True)):  # Only successful requests
                                    input_tokens.append(token_record['input_tokens'])
                                    output_tokens.append(token_record['output_tokens'])
                                    response_times.append(token_record.get('response_time_ms', 0))
                        
                        if input_tokens and response_times:
                            # Calculate success-only metrics
                            response_times.sort()
                            n = len(response_times)
                            
                            metrics['success_count'] = n
                            metrics['success_avg_response_time'] = sum(response_times) / n
                            metrics['success_min_response_time'] = min(response_times)
                            metrics['success_max_response_time'] = max(response_times)
                            metrics['success_p50'] = response_times[int(n * 0.5)]
                            metrics['success_p95'] = response_times[int(n * 0.95)] if n > 1 else response_times[0]
                            metrics['success_p99'] = response_times[int(n * 0.99)] if n > 1 else response_times[0]
                            
                            metrics['avg_input_tokens'] = sum(input_tokens) / len(input_tokens)
                            metrics['avg_output_tokens'] = sum(output_tokens) / len(output_tokens)
                            metrics['total_input_tokens'] = sum(input_tokens)
                            metrics['total_output_tokens'] = sum(output_tokens)
                        else:
                            metrics['success_count'] = 0
                            metrics['success_avg_response_time'] = 0
                            metrics['success_min_response_time'] = 0
                            metrics['success_max_response_time'] = 0
                            metrics['success_p50'] = 0
                            metrics['success_p95'] = 0
                            metrics['success_p99'] = 0
                            metrics['success_count'] = 0
                            metrics['success_avg_response_time'] = 0
                            metrics['success_min_response_time'] = 0
                            metrics['success_max_response_time'] = 0
                            metrics['success_p50'] = 0
                            metrics['success_p95'] = 0
                            metrics['success_p99'] = 0
                            metrics['avg_input_tokens'] = 0
                            metrics['avg_output_tokens'] = 0
                            metrics['total_input_tokens'] = 0
                            metrics['total_output_tokens'] = 0
                    except Exception as e:
                        print(f"Could not read token stats: {e}")
                        metrics['success_count'] = 0
                        metrics['success_avg_response_time'] = 0
                        metrics['success_min_response_time'] = 0
                        metrics['success_max_response_time'] = 0
                        metrics['success_p50'] = 0
                        metrics['success_p95'] = 0
                        metrics['success_p99'] = 0
                        metrics['avg_input_tokens'] = 0
                        metrics['avg_output_tokens'] = 0
                        metrics['total_input_tokens'] = 0
                        metrics['total_output_tokens'] = 0
                else:
                    metrics['success_count'] = 0
                    metrics['success_avg_response_time'] = 0
                    metrics['success_min_response_time'] = 0
                    metrics['success_max_response_time'] = 0
                    metrics['success_p50'] = 0
                    metrics['success_p95'] = 0
                    metrics['success_p99'] = 0
                    metrics['avg_input_tokens'] = 0
                    metrics['avg_output_tokens'] = 0
                    metrics['total_input_tokens'] = 0
                    metrics['total_output_tokens'] = 0
                
                return metrics
        
        print(f"Warning: Could not parse stats from {stats_file}")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error running test {test_id}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def generate_comparison_charts(results_df, region_prefixes, service_tiers, prompt_sizes, charts_dir):
    """Generate matplotlib charts comparing latency metrics across permutations."""
    
    print("\nGenerating comparison charts...")
    
    # 1. Chart: Average Latency by Service Tier (grouped by region and prompt size)
    fig, axes = plt.subplots(len(region_prefixes), 3, figsize=(18, 6 * len(region_prefixes)))
    if len(region_prefixes) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Average Response Time by Configuration', fontsize=16, fontweight='bold')
    
    for idx, region in enumerate(region_prefixes):
        for jdx, prompt_size in enumerate(prompt_sizes):
            ax = axes[idx][jdx]
            
            # Filter data
            data = results_df[
                (results_df['region_prefix'] == region) & 
                (results_df['prompt_size'] == prompt_size)
            ]
            
            if not data.empty:
                # Group by service tier and users
                pivot = data.pivot_table(
                    values='avg_response_time',
                    index='service_tier',
                    columns='users',
                    aggfunc='mean'
                )
                
                pivot.plot(kind='bar', ax=ax, rot=0)
                ax.set_title(f'{region.upper()} - {prompt_size.capitalize()}')
                ax.set_ylabel('Avg Response Time (ms)')
                ax.set_xlabel('Service Tier')
                ax.legend(title='Users', loc='upper left')
                ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'avg_latency_by_config.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {charts_dir / 'avg_latency_by_config.png'}")
    
    # 2. Chart: P50 and P95 Comparison across Service Tiers
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('P50 vs P95 Latency by Prompt Size', fontsize=16, fontweight='bold')
    
    for idx, prompt_size in enumerate(prompt_sizes):
        ax = axes[idx]
        
        data = results_df[results_df['prompt_size'] == prompt_size]
        
        if not data.empty:
            # Group by service tier and region
            grouped = data.groupby(['service_tier', 'region_prefix']).agg({
                'p50': 'mean',
                'p95': 'mean'
            }).reset_index()
            
            x = range(len(grouped))
            width = 0.35
            
            bars1 = ax.bar([i - width/2 for i in x], grouped['p50'], width, label='P50', alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], grouped['p95'], width, label='P95', alpha=0.8)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Response Time (ms)')
            ax.set_title(f'{prompt_size.capitalize()} Prompts')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{row['service_tier']}\n({row['region_prefix']})" 
                                for _, row in grouped.iterrows()], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'p50_p95_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {charts_dir / 'p50_p95_comparison.png'}")
    
    # 3. Chart: Latency by Prompt Size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by prompt size and service tier
    pivot = results_df.pivot_table(
        values=['avg_response_time', 'p50', 'p95'],
        index='prompt_size',
        columns='service_tier',
        aggfunc='mean'
    )
    
    x = range(len(prompt_sizes))
    width = 0.25
    
    for idx, metric in enumerate(['avg_response_time', 'p50', 'p95']):
        if metric in pivot.columns.levels[0]:
            offset = (idx - 1) * width
            for jdx, tier in enumerate(service_tiers):
                if tier in pivot[metric].columns:
                    values = [pivot[metric][tier].get(size, 0) for size in prompt_sizes]
                    ax.bar([i + offset + jdx*0.08 for i in x], values, 
                          width=0.08, label=f'{metric}-{tier}', alpha=0.8)
    
    ax.set_xlabel('Prompt Size')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Latency Metrics by Prompt Size and Service Tier')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in prompt_sizes])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'latency_by_prompt_size.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {charts_dir / 'latency_by_prompt_size.png'}")
    
    # 4. Chart: Throughput (Requests/sec) Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot = results_df.pivot_table(
        values='requests_per_sec',
        index=['service_tier', 'region_prefix'],
        columns='users',
        aggfunc='mean'
    )
    
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('Throughput (Requests/sec) by Configuration and User Count', fontsize=14, fontweight='bold')
    ax.set_ylabel('Requests per Second')
    ax.set_xlabel('Service Tier & Region')
    ax.legend(title='Users', loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {charts_dir / 'throughput_comparison.png'}")
    
    # 5. Heatmap: Average Latency across all dimensions
    fig, axes = plt.subplots(len(region_prefixes), 1, figsize=(14, 5 * len(region_prefixes)))
    if len(region_prefixes) == 1:
        axes = [axes]
    
    fig.suptitle('Latency Heatmap: Service Tier vs Prompt Size', fontsize=16, fontweight='bold')
    
    for idx, region in enumerate(region_prefixes):
        ax = axes[idx]
        
        # Create pivot table for heatmap
        data = results_df[results_df['region_prefix'] == region]
        pivot = data.pivot_table(
            values='avg_response_time',
            index='service_tier',
            columns='prompt_size',
            aggfunc='mean'
        )
        
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([s.capitalize() for s in pivot.columns])
        ax.set_yticklabels(pivot.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Avg Response Time (ms)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(f'{region.upper()} Region')
        ax.set_xlabel('Prompt Size')
        ax.set_ylabel('Service Tier')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'latency_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {charts_dir / 'latency_heatmap.png'}")
    
    print("\nAll charts generated successfully!")


def main():
    """Main orchestration function."""
    args = parse_arguments()
    
    # Parse comma-separated lists
    region_prefixes = [x.strip() for x in args.region_prefixes.split(',')]
    service_tiers = [x.strip() for x in args.service_tiers.split(',')]
    prompt_sizes = [x.strip() for x in args.prompt_sizes.split(',')]
    user_counts = [int(x.strip()) for x in args.user_counts.split(',')]
    
    results_dir = args.results_dir
    charts_dir = results_dir / 'charts'
    
    print("="*80)
    print("BEDROCK LOAD TEST ORCHESTRATION")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Base Model ID: {args.model_id}")
    print(f"  Region Prefixes: {region_prefixes}")
    print(f"  Service Tiers: {service_tiers}")
    print(f"  Prompt Sizes: {prompt_sizes}")
    print(f"  User Counts: {user_counts}")
    print(f"  Test Duration: {args.test_duration}")
    print(f"  Spawn Rate: {args.spawn_rate} users/sec")
    print(f"  Results Directory: {results_dir}")
    print(f"  Include Failures: {args.include_failures}")
    print()
    
    # Setup directories
    setup_directories(results_dir)
    
    # Set environment variable for locustfile
    os.environ['INCLUDE_FAILURES'] = str(args.include_failures).lower()
    
    # Generate all permutations
    permutations = list(product(region_prefixes, service_tiers, prompt_sizes, user_counts))
    total_tests = len(permutations)
    
    print(f"Total test permutations: {total_tests}")
    print(f"Estimated total time: ~{total_tests * int(args.test_duration[:-1])} minutes\n")
    
    # Confirm before proceeding
    if not args.yes:
        response = input("Proceed with tests? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Test execution cancelled.")
            return
    
    # Run all tests
    results = []
    start_time = time.time()
    
    for test_num, (region, tier, prompt, users) in enumerate(permutations, 1):
        test_id = f"{region}_{tier}_{prompt}_u{users}"
        
        print(f"\nProgress: {test_num}/{total_tests}")
        
        result = run_locust_test(
            model_id=args.model_id,
            region_prefix=region,
            service_tier=tier,
            prompt_size=prompt,
            users=users,
            spawn_rate=args.spawn_rate,
            test_id=test_id,
            results_dir=results_dir,
            test_duration=args.test_duration
        )
        
        if result:
            results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"All tests completed in {total_time/60:.2f} minutes")
    print(f"{'='*80}\n")
    
    # Save consolidated results
    if results:
        results_df = pd.DataFrame(results)
        results_file = results_dir / f'consolidated_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Consolidated results saved to: {results_file}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print("\nAverage Response Time by Service Tier:")
        print(results_df.groupby('service_tier')['avg_response_time'].mean().to_string())
        
        print("\nAverage Response Time by Prompt Size:")
        print(results_df.groupby('prompt_size')['avg_response_time'].mean().to_string())
        
        print("\nAverage Response Time by Region:")
        print(results_df.groupby('region_prefix')['avg_response_time'].mean().to_string())
        
        print("\nP95 Latency by Service Tier:")
        print(results_df.groupby('service_tier')['p95'].mean().to_string())
        
        print("\nFailure Rate by Configuration:")
        results_df['failure_rate'] = (results_df['failure_count'] / results_df['total_requests'] * 100).round(2)
        print(results_df.groupby(['region_prefix', 'service_tier', 'prompt_size'])['failure_rate'].mean().to_string())
        
        print("\nAverage Input Tokens by Prompt Size:")
        print(results_df.groupby('prompt_size')['avg_input_tokens'].mean().to_string())
        
        print("\nAverage Output Tokens by Service Tier:")
        print(results_df.groupby('service_tier')['avg_output_tokens'].mean().to_string())
        
        # Generate charts
        generate_comparison_charts(results_df, region_prefixes, service_tiers, prompt_sizes, charts_dir)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {results_dir}")
        print(f"Charts saved to: {charts_dir}")
        print(f"{'='*80}\n")
    else:
        print("No results to process.")


if __name__ == '__main__':
    main()