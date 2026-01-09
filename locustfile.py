from locust import User, task, between
import logging
import boto3
import os
import json
import time
from pathlib import Path
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration from environment
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', 'test_results'))
INCLUDE_FAILURES = os.getenv('INCLUDE_FAILURES', 'false').lower() == 'true'


class BedrockUser(User):
    wait_time = between(1, 2)
    
    def on_start(self):
        """Initialize Bedrock client and load configuration."""
        # Configuration from environment
        base_model_id = os.getenv('BASE_MODEL_ID', 'amazon.nova-2-lite-v1:0')
        region_prefix = os.getenv('REGION_PREFIX', 'us')
        self.service_tier = os.getenv('SERVICE_TIER')
        self.prompt_size = os.getenv('PROMPT_SIZE', 'small')
        
        # Build full model ID
        self.model_id = f"{region_prefix}.{base_model_id}"
        
        # Load prompt
        with open(os.getenv('PROMPTS_FILE', 'prompts.json'), 'r') as f:
            prompts = json.load(f)
        prompt_text = prompts[self.prompt_size]['text']
        
        # Prepare request payload
        self.messages = [{"role": "user", "content": [{"text": prompt_text}]}]
        max_tokens = int(os.getenv('MAX_TOKENS', '4096'))
        self.inference_config = {"temperature": 0.5, "maxTokens": max_tokens, "topP": 1}
        
        # Create Bedrock client
        config = Config(connect_timeout=840, read_timeout=840)
        self.bedrock = boto3.client('bedrock-runtime', config=config)
        
        # Request name for metrics
        self.request_name = f"[{region_prefix}][{self.service_tier or 'none'}][{self.prompt_size}]"
        
        logger.info(f"Initialized: {self.model_id}, tier={self.service_tier}, prompt={self.prompt_size}")
    
    @task
    def converse_request(self):
        """Execute Bedrock Converse API request."""
        start_time = time.time()
        
        try:
            # Build request parameters
            params = {
                "modelId": self.model_id,
                "messages": self.messages,
                "inferenceConfig": self.inference_config
            }
            if self.service_tier:
                params["serviceTier"] = {"type": self.service_tier}
            
            # Call Bedrock API
            response = self.bedrock.converse(**params)
            response_time = (time.time() - start_time) * 1000
            
            # Extract token usage
            tokens = response['usage']
            logger.info(f"Tokens - Input: {tokens['inputTokens']}, Output: {tokens['outputTokens']}")
            
            # Save token data
            token_file = RESULTS_DIR / 'token_data.jsonl'
            token_file.parent.mkdir(exist_ok=True)
            with open(token_file, 'a') as f:
                f.write(json.dumps({
                    'timestamp': time.time(),
                    'region_prefix': os.getenv('REGION_PREFIX', 'us'),
                    'service_tier': self.service_tier or 'none',
                    'prompt_size': self.prompt_size,
                    'response_time_ms': response_time,
                    'input_tokens': tokens['inputTokens'],
                    'output_tokens': tokens['outputTokens'],
                    'total_tokens': tokens['totalTokens']
                }) + '\n')
            
            # Record success
            self.environment.events.request.fire(
                request_type="Converse",
                name=self.request_name,
                response_time=response_time,
                response_length=0,
                exception=None,
                context={}
            )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Request failed: {e}")
            
            # Record failure (response_time=None excludes from latency metrics unless INCLUDE_FAILURES=true)
            self.environment.events.request.fire(
                request_type="Converse",
                name=self.request_name,
                response_time=response_time if INCLUDE_FAILURES else None,
                response_length=0,
                exception=e,
                context={}
            )
