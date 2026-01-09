# Bedrock Load Testing with Locust

This project provides a comprehensive load testing framework for AWS Bedrock Converse API with automated permutation testing and latency visualization.

## Overview

The testing framework evaluates Bedrock model performance across multiple dimensions:
- **Region Prefixes**: `us.` (see [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html) for region and model compatibility)
- **Service Tiers**: `default`, `priority`, and `flex`
- **Prompt Sizes**: Small (~5 words), Medium (~40 words), and Large (~150 words)
- **User Loads**: Configurable concurrent users (default: 30, 60, 90)

## Files

- `locustfile.py` - Locust test definition with configurable parameters
- `run_tests.py` - Orchestration script that runs all permutations and generates charts
- `prompts.json` - Sample prompts of varying sizes
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

3. (Optional) Set custom model ID:
```bash
export BASE_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Usage

### Run Full Test Suite

Execute all permutations with default settings:

```bash
python run_tests.py
```

Or customize the test run with command-line arguments:

```bash
# Test specific model with custom results directory
python run_tests.py --model-id amazon.nova-2-lite-v1:0 --results-dir test_results/nova_run

# Test only specific configurations
python run_tests.py --region-prefixes us --service-tiers priority,flex --user-counts 30,60

# Include failures in latency metrics
python run_tests.py --include-failures

# Quick test with shorter duration
python run_tests.py --test-duration 30s --user-counts 10

# Skip confirmation prompt
python run_tests.py --yes
```

View all available options:
```bash
python run_tests.py --help
```

This will:
- Run tests for all combinations of region prefixes, service tiers, prompt sizes, and user counts
- Generate CSV files with detailed metrics for each test
- Create HTML reports for each test run
- Generate comparison charts showing P50, P95, and average latencies
- Save all results to the specified directory

### Run Individual Tests

Test a specific configuration using Locust directly:

```bash
# Set environment variables
export BASE_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
export REGION_PREFIX=us
export SERVICE_TIER=priority
export PROMPT_SIZE=medium

# Run Locust
locust -f locustfile.py --headless --users 30 --spawn-rate 10 --run-time 5m
```

### Command-Line Options

The `run_tests.py` script accepts the following arguments:

- `--model-id` - Base Bedrock model ID without region prefix (default: `amazon.nova-premier-v1:0`)
- `--results-dir` - Directory to store test results (default: `test_results`)
- `--region-prefixes` - Comma-separated region prefixes: `us` (default: `us`). See [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html) for region and model compatibility
- `--service-tiers` - Comma-separated service tiers: `default`, `priority`, `flex` (default: all)
- `--prompt-sizes` - Comma-separated prompt sizes: `small`, `medium`, `large` (default: all)
- `--user-counts` - Comma-separated concurrent user counts (default: `30,60,90`)
- `--test-duration` - Duration for each test: `30s`, `1m`, `5m`, etc. (default: `1m`)
- `--spawn-rate` - Users spawned per second (default: `10`)
- `--max-tokens` - Maximum output tokens for model responses (default: `4096`)
- `--include-failures` - Include failed requests in latency metrics (default: excluded)
- `--yes`, `-y` - Skip confirmation prompt

### Environment Variables (Alternative)

You can also use environment variables instead of command-line arguments:

### Environment Variables (Alternative)

You can also use environment variables instead of command-line arguments:

```bash
export BASE_MODEL_ID=amazon.nova-2-lite-v1:0
export RESULTS_DIR=test_results/baseline
export REGION_PREFIXES=us
export USER_COUNTS=30,60,90
export MAX_TOKENS=2048
export INCLUDE_FAILURES=true
python run_tests.py
```

Environment variables for `locustfile.py` (when running Locust directly):

- `BASE_MODEL_ID` - Base model ID without region prefix
- `REGION_PREFIX` - Region prefix (e.g., `us`). See [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html) for region and model compatibility
- `SERVICE_TIER` - Either `default`, `priority`, or `flex`
- `PROMPT_SIZE` - Either `small`, `medium`, or `large`
- `PROMPTS_FILE` - Path to prompts JSON file (default: `prompts.json`)
- `RESULTS_DIR` - Directory for test results (default: `test_results`)
- `MAX_TOKENS` - Maximum output tokens (default: `4096`)
- `INCLUDE_FAILURES` - Include failed requests in latency metrics (default: `false`)

## Test Permutations

By default, the framework tests:
- 1 region prefix × 3 service tiers × 3 prompt sizes × 3 user counts = **27 test permutations**

Each test runs for 1 minute, so the full suite takes approximately **27 minutes** to complete.

**Note**: Different models are available in different regions with different service tiers. Refer to the [AWS Bedrock service tiers documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html) for region and model compatibility details.

## Output

### CSV Reports
Individual test results are saved as:
- `{RESULTS_DIR}/test_{config}_stats.csv` - Detailed statistics
- `{RESULTS_DIR}/consolidated_results_{timestamp}.csv` - All results combined
- `{RESULTS_DIR}/token_data.jsonl` - Per-request token usage and response times

### HTML Reports
Interactive HTML reports for each test:
- `{RESULTS_DIR}/test_{config}_report.html`

### Charts
Comparison charts saved to `{RESULTS_DIR}/charts/`:
- `avg_latency_by_config.png` - Average latency by all configurations
- `p50_p95_comparison.png` - P50 vs P95 latencies by prompt size
- `latency_by_prompt_size.png` - Latency metrics across prompt sizes
- `throughput_comparison.png` - Requests/sec by configuration
- `latency_heatmap.png` - Heatmap of service tier vs prompt size

### Organizing Multiple Test Runs

To keep results from different test runs separate, use the `RESULTS_DIR` variable:

```bash
# First run with baseline configuration
export RESULTS_DIR=test_results/baseline_run
python run_tests.py

# Second run with different model
export RESULTS_DIR=test_results/nova_run
export BASE_MODEL_ID=amazon.nova-2-lite-v1:0
python run_tests.py

# Third run including failures in metrics
export RESULTS_DIR=test_results/with_failures
export INCLUDE_FAILURES=true
python run_tests.py
```

## Metrics Collected

For each test configuration, the framework tracks:
- Total requests and failures (failures excluded from latency by default)
- Average, min, max response times
- P50, P95, P99 latencies
- Requests per second (throughput)
- Token usage (input, output, total per request in token_data.jsonl)

**Note on Failure Handling**: By default, failed requests are excluded from latency metrics to give you accurate success-only performance. To include failures in latency calculations, set `INCLUDE_FAILURES=true`.

## Customizing Prompts

Edit `prompts.json` to use your own prompts:

```json
{
  "small": {
    "text": "Your short prompt here",
    "description": "Description"
  },
  "medium": {
    "text": "Your medium-length prompt here",
    "description": "Description"
  },
  "large": {
    "text": "Your large prompt here",
    "description": "Description"
  }
}
```

## Example: Quick Test

For a quick test with fewer permutations:

```bash
# Test only US region with one user count
export USER_COUNTS=30

# Modify run_tests.py to test fewer combinations
# Or run individual tests manually
```

## Troubleshooting

- **Timeout errors**: Increase timeouts in locustfile.py (currently 840s)
- **Rate limiting**: Add delays between tests or reduce concurrent users
- **Missing metrics**: Check that CSV files are being generated in test_results/
- **Chart generation errors**: Ensure matplotlib is installed correctly

## Notes

- The locustfile uses the Converse API without tool use
- Service tiers are passed via the `serviceTier` parameter
- Region prefixes are prepended to the model ID
- Tests include proper error handling and logging
