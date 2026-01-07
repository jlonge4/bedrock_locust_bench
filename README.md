# Bedrock Load Testing with Locust

This project provides a comprehensive load testing framework for AWS Bedrock Converse API with automated permutation testing and latency visualization.

## Overview

The testing framework evaluates Bedrock model performance across multiple dimensions:
- **Region Prefixes**: `us.` and `global.`
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

Execute all permutations and generate comparison charts:

```bash
python run_tests.py
```

This will:
- Run tests for all combinations of region prefixes, service tiers, prompt sizes, and user counts
- Generate CSV files with detailed metrics for each test
- Create HTML reports for each test run
- Generate comparison charts showing P50, P95, and average latencies
- Save all results to `test_results/` directory

### Run Individual Tests

Test a specific configuration:

```bash
# Set environment variables
export BASE_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
export REGION_PREFIX=us
export SERVICE_TIER=priority
export PROMPT_SIZE=medium

# Run Locust
locust -f locustfile.py --headless --users 30 --spawn-rate 10 --run-time 5m
```

### Configuration Options

Environment variables for `run_tests.py`:

- `BASE_MODEL_ID` - Base model ID without region prefix (default: `anthropic.claude-3-5-sonnet-20241022-v2:0`)
- `USER_COUNTS` - Comma-separated list of user counts (default: `30,60,90`)

Environment variables for `locustfile.py`:

- `BASE_MODEL_ID` - Base model ID without region prefix
- `REGION_PREFIX` - Either `us` or `global`
- `SERVICE_TIER` - Either `default`, `priority`, or `flex`
- `PROMPT_SIZE` - Either `small`, `medium`, or `large`
- `PROMPTS_FILE` - Path to prompts JSON file (default: `prompts.json`)

## Test Permutations

By default, the framework tests:
- 2 region prefixes × 3 service tiers × 3 prompt sizes × 3 user counts = **54 test permutations**

Each test runs for 5 minutes, so the full suite takes approximately **4.5 hours** to complete.

## Output

### CSV Reports
Individual test results are saved as:
- `test_results/test_{config}_stats.csv` - Detailed statistics
- `test_results/consolidated_results_{timestamp}.csv` - All results combined

### HTML Reports
Interactive HTML reports for each test:
- `test_results/test_{config}_report.html`

### Charts
Comparison charts saved to `test_results/charts/`:
- `avg_latency_by_config.png` - Average latency by all configurations
- `p50_p95_comparison.png` - P50 vs P95 latencies by prompt size
- `latency_by_prompt_size.png` - Latency metrics across prompt sizes
- `throughput_comparison.png` - Requests/sec by configuration
- `latency_heatmap.png` - Heatmap of service tier vs prompt size

## Metrics Collected

For each test configuration, the framework tracks:
- Total requests and failures
- Average, min, max response times
- P50, P95, P99 latencies
- Requests per second (throughput)
- Token usage (logged)

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
