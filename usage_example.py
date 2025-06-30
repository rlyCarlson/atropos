#!/usr/bin/env python3
"""
Usage Examples for Conversational Style DPO Environments

This script demonstrates how to use the conversational style DPO environments
for training models to prefer more engaging, empathetic, and helpful responses
over blunt or unhelpful ones.

There are two main environments:
1. ConversationalStyleDPOEnv - Uses synthetic data with predefined prompt/response pairs
2. GSM8KConversationalStyleDPOEnv - Dynamically generates prompts and responses using LLM

Requirements:
- Install dependencies: pip install -r requirements.txt
- Set up your API keys if using the GSM8K environment
"""

import asyncio
import logging
import os
from typing import List, Optional

# Import the environments
from conversational_style_dpo_env import (
    ConversationalStyleDPOEnv,
    ConversationalStyleDPOEnvConfig,
)
from gsmk8k_conversational_style_dpo_env import (
    GSM8KConversationalStyleDPOEnv,
    GSM8KConversationalStyleDPOEnvConfig,
)

# Import Atropos utilities
from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.server_handling.server_baseline import ServerBaseline
# from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer_dpo
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_environment():
    """
    Example 1: Using the basic ConversationalStyleDPOEnv with synthetic data
    This environment uses predefined prompt/response pairs and doesn't require API calls.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Conversational Style DPO Environment")
    print("="*60)
    
    # Create configuration
    config = ConversationalStyleDPOEnvConfig(
        dataset_name="synthetic_conversational_style",
        shuffle_dataset=True
    )
    
    # Initialize environment (use ServerBaseline instead of empty list)
    env = ConversationalStyleDPOEnv(
        config=config,
        server_configs=ServerBaseline(),  # Use ServerBaseline instead of empty list
        testing=True
    )
    
    # Setup the environment
    await env.setup()
    print(f"Environment setup complete. Dataset has {len(env.dataset)} examples.")
    
    # Get a few examples from the dataset
    print("\nSample data from the environment:")
    for i in range(3):
        item = await env.get_next_item()
        prompt, chosen, rejected = item
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Chosen (preferred): {chosen}")
        print(f"Rejected (not preferred): {rejected}")
    
    # Demonstrate trajectory collection (for training)
    print("\nDemonstrating trajectory collection:")
    item = await env.get_next_item()
    scored_group, items = await env.collect_trajectories([item])
    
    if scored_group:
        print(f"Successfully created ScoredDataGroup with {len(scored_group.data)} items")
        print(f"Sample data: {scored_group.data[0] if scored_group.data else 'No data'}")
    else:
        print("Failed to create ScoredDataGroup")


async def example_gsm8k_environment():
    """
    Example 2: Using the GSM8K ConversationalStyleDPOEnv with dynamic generation
    This environment generates prompts and responses using an LLM API.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: GSM8K Conversational Style DPO Environment")
    print("="*60)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("This example requires an OpenAI API key to generate dynamic prompts and responses.")
        print("Set your API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Create server configuration for OpenAI
    server_config = APIServerConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model="gpt-3.5-turbo"  # You can change this to other models
    )
    
    # Create environment configuration
    config = GSM8KConversationalStyleDPOEnvConfig(
        dataset_name="dynamic_conversational_style",
        shuffle_dataset=True,
        chosen_temperature=0.7,
        chosen_max_tokens=150,
        rejected_temperature=0.4,
        rejected_max_tokens=50,
        prompt_generation_temperature=0.8,
        prompt_generation_max_tokens=1000
    )
    
    # Initialize environment
    env = GSM8KConversationalStyleDPOEnv(
        config=config,
        server_configs=[server_config],
        testing=True
    )
    
    # Setup the environment (this will generate prompts via LLM)
    print("Setting up environment and generating prompts via LLM...")
    await env.setup()
    print(f"Environment setup complete. Generated {len(env.prompt_dataset)} prompts.")
    
    # Get a few examples
    print("\nSample dynamically generated data:")
    for i in range(2):  # Show fewer examples as they're more expensive
        item = await env.get_next_item()
        prompt, chosen, rejected = item
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Chosen (preferred): {chosen}")
        print(f"Rejected (not preferred): {rejected}")
    
    # Demonstrate trajectory collection
    print("\nDemonstrating trajectory collection with dynamic data:")
    item = await env.get_next_item()
    scored_group, items = await env.collect_trajectories([item])
    
    if scored_group:
        print(f"Successfully created ScoredDataGroup with {len(scored_group.data)} items")
    else:
        print("Failed to create ScoredDataGroup")


async def example_batch_processing():
    """
    Example 3: Processing multiple items in batch for training
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing for Training")
    print("="*60)
    
    # Use the basic environment for this example
    config = ConversationalStyleDPOEnvConfig(
        dataset_name="synthetic_conversational_style",
        shuffle_dataset=True
    )
    
    env = ConversationalStyleDPOEnv(
        config=config,
        server_configs=[],
        testing=True
    )
    
    await env.setup()
    
    # Collect multiple items for batch processing
    batch_size = 3
    items = []
    
    print(f"Collecting {batch_size} items for batch processing...")
    for _ in range(batch_size):
        item = await env.get_next_item()
        items.append(item)
    
    # Process the batch
    scored_group, processed_items = await env.collect_trajectories(items)
    
    if scored_group:
        print(f"Successfully processed batch of {len(items)} items")
        print(f"Created ScoredDataGroup with {len(scored_group.data)} training examples")
        
        # Show what the training data looks like
        if scored_group.data:
            sample = scored_group.data[0]
            print(f"\nSample training data structure:")
            print(f"Keys: {list(sample.keys())}")
            if 'chosen_input_ids' in sample:
                print(f"Chosen sequence length: {len(sample['chosen_input_ids'])}")
            if 'rejected_input_ids' in sample:
                print(f"Rejected sequence length: {len(sample['rejected_input_ids'])}")
    else:
        print("Failed to process batch")


def example_configuration_options():
    """
    Example 4: Show different configuration options
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration Options")
    print("="*60)
    
    print("Basic Environment Configuration Options:")
    basic_config = ConversationalStyleDPOEnvConfig()
    print(f"- dataset_name: {basic_config.dataset_name}")
    print(f"- shuffle_dataset: {basic_config.shuffle_dataset}")
    
    print("\nGSM8K Environment Configuration Options:")
    gsm8k_config = GSM8KConversationalStyleDPOEnvConfig()
    print(f"- dataset_name: {gsm8k_config.dataset_name}")
    print(f"- shuffle_dataset: {gsm8k_config.shuffle_dataset}")
    print(f"- chosen_temperature: {gsm8k_config.chosen_temperature}")
    print(f"- chosen_max_tokens: {gsm8k_config.chosen_max_tokens}")
    print(f"- rejected_temperature: {gsm8k_config.rejected_temperature}")
    print(f"- rejected_max_tokens: {gsm8k_config.rejected_max_tokens}")
    print(f"- prompt_generation_temperature: {gsm8k_config.prompt_generation_temperature}")
    print(f"- prompt_generation_max_tokens: {gsm8k_config.prompt_generation_max_tokens}")
    print(f"- data_path_to_save_groups: {gsm8k_config.data_path_to_save_groups}")


async def example_custom_dataset():
    """
    Example 5: Creating a custom dataset for the basic environment
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Dataset Creation")
    print("="*60)
    
    # You can extend the basic environment to use your own data
    # by modifying the setup method or creating a subclass
    
    class CustomConversationalStyleDPOEnv(ConversationalStyleDPOEnv):
        async def setup(self):
            """Override setup to use custom data."""
            # Your custom dataset
            self.synthetic_data = [
                {
                    "prompt": "How do I make a good cup of coffee?",
                    "chosen": "Great question! Start with fresh, quality beans and grind them just before brewing. Use filtered water heated to 195-205°F. For a pour-over, use a 1:16 coffee-to-water ratio. What brewing method do you prefer?",
                    "rejected": "Put coffee in hot water."
                },
                {
                    "prompt": "I'm nervous about my job interview tomorrow.",
                    "chosen": "It's completely normal to feel nervous! Preparation is key - review common questions, research the company, and practice your responses. Remember to breathe and be yourself. What aspect are you most concerned about?",
                    "rejected": "Good luck."
                },
                {
                    "prompt": "What's the best way to learn programming?",
                    "chosen": "Start with a beginner-friendly language like Python and work on small projects that interest you. Practice regularly, join coding communities, and don't be afraid to make mistakes - they're part of learning! What kind of projects excite you?",
                    "rejected": "Read a book."
                }
            ]
            
            self.dataset = self.synthetic_data
            if self.config.shuffle_dataset:
                import random
                random.shuffle(self.dataset)
            self.iter = 0
            logger.info(f"Loaded custom dataset with {len(self.dataset)} examples.")
    
    # Use the custom environment
    config = ConversationalStyleDPOEnvConfig(
        dataset_name="custom_conversational_style",
        shuffle_dataset=True
    )
    
    env = CustomConversationalStyleDPOEnv(
        config=config,
        server_configs=[],
        testing=True
    )
    
    await env.setup()
    print(f"Custom environment setup complete. Dataset has {len(env.dataset)} examples.")
    
    # Show the custom data
    print("\nCustom dataset examples:")
    for i, item in enumerate(env.dataset):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {item['prompt']}")
        print(f"Chosen: {item['chosen']}")
        print(f"Rejected: {item['rejected']}")


async def main():
    """
    Run all examples
    """
    print("Conversational Style DPO Environment Usage Examples")
    print("="*60)
    
    try:
        # Example 1: Basic environment
        await example_basic_environment()
        
        # Example 2: GSM8K environment (requires API key)
        # await example_gsm8k_environment()
        
        # Example 3: Batch processing
        await example_batch_processing()
        
        # Example 4: Configuration options
        example_configuration_options()
        
        # Example 5: Custom dataset
        await example_custom_dataset()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 