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


def generate_conversation(bedrock_client, model_id, messages, service_tier=None):
    """
    Sends messages to a model using the Converse API.
    
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        messages (list): The messages to send to the model.
        service_tier (str): The service tier to use ('priority', 'default', 'flex', or None)

    Returns:
        response (dict): The conversation that the model generated.
    """

    logger.info("Generating message with model %s", model_id)

    # Inference parameters to use.
    temperature = 0.5
    max_tokens = 4096

    # Base inference parameters to use.
    inference_config = {
        "temperature": temperature,
        "maxTokens": max_tokens,
        "topP": 1,
    }
    
    # Prepare converse parameters
    converse_params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config,
    }
    
    # Add service tier if specified
    if service_tier:
        converse_params["serviceTier"] = {"type": service_tier}
    
    # Send the message.
    response = bedrock_client.converse(**converse_params)

    # Log token usage.
    token_usage = response['usage']
    logger.info("Input tokens: %s", token_usage['inputTokens'])
    logger.info("Output tokens: %s", token_usage['outputTokens'])
    logger.info("Total tokens: %s", token_usage['totalTokens'])
    logger.info("Stop reason: %s", response['stopReason'])

    return response, token_usage


class BedrockUser(User):
    wait_time = between(1, 2)
    
    def on_start(self):
        """Called when a simulated user starts running."""
        # Get configuration from environment variables
        self.base_model_id = os.getenv('BASE_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        self.region_prefix = os.getenv('REGION_PREFIX', 'us')  # 'us' or 'global'
        self.service_tier = os.getenv('SERVICE_TIER')  # 'priority', 'default', 'flex', or None
        self.prompt_size = os.getenv('PROMPT_SIZE', 'medium')  # 'small', 'medium', or 'large'
        
        # Construct full model ID with region prefix
        self.model_id = f"{self.region_prefix}.{self.base_model_id}"
        
        # Load prompts from file
        prompts_file = os.getenv('PROMPTS_FILE', 'prompts.json')
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        # Get the prompt text for the specified size
        self.prompt_text = prompts[self.prompt_size]['text']
        
        # Prepare messages
        self.messages = [
            {
                "role": "user",
                "content": [{"text": self.prompt_text}],
            }
        ]
        
        # Configure Bedrock client with extended timeouts
        custom_config = Config(connect_timeout=840, read_timeout=840)
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            config=custom_config
        )
        
        logger.info(f"Initialized user with model: {self.model_id}, "
                   f"service tier: {self.service_tier}, "
                   f"prompt size: {self.prompt_size}")
    
    @task
    def converse_request(self):
        """Execute a Converse API request."""
        # Create a descriptive name for the request
        request_name = f"[{self.region_prefix}][{self.service_tier or 'none'}][{self.prompt_size}]"
        
        with self.environment.events.request.measure("Converse", request_name):
            response, token_usage = generate_conversation(
                self.bedrock_client,
                self.model_id,
                self.messages,
                self.service_tier
            )
            
            logger.debug(f"Response: {response['output']['message']['content'][0]['text'][:100]}...")
            
            # Write token data to file immediately after each successful request
            # (only successful requests reach here since exceptions bubble up)
            token_file = Path('test_results') / 'token_data.jsonl'
            token_file.parent.mkdir(exist_ok=True)
            
            token_record = {
                'timestamp': time.time(),
                'region_prefix': self.region_prefix,
                'service_tier': self.service_tier or 'none',
                'prompt_size': self.prompt_size,
                'input_tokens': token_usage['inputTokens'],
                'output_tokens': token_usage['outputTokens'],
                'total_tokens': token_usage['totalTokens']
            }
            
            # Append to JSONL file (one JSON object per line)
            with open(token_file, 'a') as f:
                f.write(json.dumps(token_record) + '\n')
