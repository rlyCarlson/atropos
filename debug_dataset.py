#!/usr/bin/env python3
"""
Debug script to test dataset loading
"""

import asyncio
import logging
import sys
import os

# Add the atropos root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from chatbot_arena_dpo_env import ChatbotArenaDPOEnvConfig, ChatbotArenaDPOEnv

logging.basicConfig(level=logging.INFO)

async def debug_dataset():
    """Debug the dataset loading process"""
    
    # Create environment config
    env_config = ChatbotArenaDPOEnvConfig(
        dataset_name="chatbot_arena_preferences",
        shuffle_dataset=True,
        data_file_path="",
        max_examples=10,  # Just load 10 examples for debugging
        use_hf_dataset=True,
        filter_ties=True,
        filter_both_bad=True
    )
    
    print("Creating environment...")
    env = ChatbotArenaDPOEnv(config=env_config, server_configs=[], testing=True)
    
    print("Setting up environment...")
    await env.setup()
    
    print(f"Dataset loaded with {len(env.dataset)} items")
    
    if len(env.dataset) > 0:
        print(f"First item: {env.dataset[0]}")
        print(f"First item type: {type(env.dataset[0])}")
        
        if isinstance(env.dataset[0], dict):
            print(f"First item keys: {list(env.dataset[0].keys())}")
            
            # Check each field
            for key in ["prompt", "chosen", "rejected"]:
                if key in env.dataset[0]:
                    value = env.dataset[0][key]
                    print(f"{key}: {type(value)} - {value}")
                else:
                    print(f"{key}: NOT FOUND")
    
    # Test getting a few items
    print("\nTesting get_next_item...")
    for i in range(3):
        try:
            item = await env.get_next_item()
            print(f"Item {i}: {item}")
        except Exception as e:
            print(f"Error getting item {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_dataset()) 