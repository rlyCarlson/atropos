#!/usr/bin/env python3
"""
Usage Example for Chatbot Arena DPO Environment

This script demonstrates how to use the Chatbot Arena DPO environment
for training models on preference data from chatbot arena results.

Requirements:
- Install dependencies: pip install -r requirements.txt
- Prepare your arena data in JSON format (see sample_arena_data.json)
- Or use the Hugging Face dataset: lmarena-ai/arena-human-preference-55k
"""

import asyncio
import logging
import os
import json
from typing import List

# Import our custom environment
from chatbot_arena_dpo_env import (
    ChatbotArenaDPOEnv,
    ChatbotArenaDPOEnvConfig,
)

# Import Atropos utilities
from atroposlib.envs.base import APIServerConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_huggingface_dataset():
    """
    Example 1: Using the ChatbotArenaDPOEnv with Hugging Face dataset
    This loads the real Chatbot Arena data from lmarena-ai/arena-human-preference-55k
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Chatbot Arena DPO Environment with Hugging Face Dataset")
    print("="*60)
    
    # Create configuration to use Hugging Face dataset
    config = ChatbotArenaDPOEnvConfig(
        dataset_name="chatbot_arena_preferences",
        shuffle_dataset=True,
        use_hf_dataset=True,  # Use Hugging Face dataset
        filter_ties=True,     # Filter out ties
        filter_both_bad=True, # Filter out cases where both responses are bad
        max_examples=1000     # Limit to 1000 examples for demo
    )
    
    # Initialize environment
    env = ChatbotArenaDPOEnv(
        config=config,
        server_configs=[],
        testing=True
    )
    
    # Setup the environment (this will download and process the HF dataset)
    print("Setting up environment and loading Hugging Face dataset...")
    await env.setup()
    print(f"Environment setup complete. Dataset has {len(env.dataset)} examples.")
    
    # Get a few examples from the dataset
    print("\nSample data from Hugging Face dataset:")
    for i in range(3):
        item = await env.get_next_item()
        prompt, chosen, rejected = item
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompt[:200]}...")  # Truncate for display
        print(f"Chosen (preferred): {chosen[:200]}...")
        print(f"Rejected (not preferred): {rejected[:200]}...")
    
    # Demonstrate trajectory collection (for training)
    print("\nDemonstrating trajectory collection:")
    item = await env.get_next_item()
    scored_group, items = await env.collect_trajectories([item])
    
    if scored_group:
        print(f"Successfully created ScoredDataGroup with {len(scored_group.data)} items")
        print(f"Sample data: {scored_group.data[0] if scored_group.data else 'No data'}")
    else:
        print("Failed to create ScoredDataGroup")


async def example_basic_environment():
    """
    Example 2: Using the ChatbotArenaDPOEnv with sample data
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Basic Chatbot Arena DPO Environment")
    print("="*60)
    
    # Create configuration with sample data
    config = ChatbotArenaDPOEnvConfig(
        dataset_name="chatbot_arena_preferences",
        shuffle_dataset=True,
        use_hf_dataset=False,  # Use sample data instead
        data_file_path="",     # Empty means use sample data
        max_examples=None
    )
    
    # Initialize environment
    env = ChatbotArenaDPOEnv(
        config=config,
        server_configs=[],
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


async def example_with_json_data():
    """
    Example 3: Using the environment with JSON data file
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Chatbot Arena DPO Environment with JSON Data")
    print("="*60)
    
    # Check if sample data file exists
    data_file = "sample_arena_data.json"
    if not os.path.exists(data_file):
        print(f"Sample data file {data_file} not found. Creating it...")
        create_sample_data_file(data_file)
    
    # Create configuration with JSON data file
    config = ChatbotArenaDPOEnvConfig(
        dataset_name="chatbot_arena_preferences",
        shuffle_dataset=True,
        use_hf_dataset=False,  # Use local JSON file
        data_file_path=data_file,
        max_examples=5  # Limit to 5 examples for demo
    )
    
    # Initialize environment
    env = ChatbotArenaDPOEnv(
        config=config,
        server_configs=[],
        testing=True
    )
    
    # Setup the environment
    await env.setup()
    print(f"Environment setup complete. Loaded {len(env.dataset)} examples from {data_file}.")
    
    # Get examples from the JSON file
    print("\nSample data from JSON file:")
    for i in range(3):
        item = await env.get_next_item()
        prompt, chosen, rejected = item
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Chosen (preferred): {chosen}")
        print(f"Rejected (not preferred): {rejected}")


def create_sample_data_file(filename: str):
    """
    Create a sample JSON data file if it doesn't exist
    """
    sample_data = [
        {
            "prompt": "What's the best way to learn machine learning?",
            "chosen": "Great question! I'd recommend starting with the fundamentals: linear algebra, calculus, and statistics. Then move to Python and libraries like scikit-learn. Start with simple projects and gradually work up to more complex ones. Online courses from Coursera, edX, or fast.ai are excellent resources. What's your current background in programming and math?",
            "rejected": "Just take a course online."
        },
        {
            "prompt": "I'm feeling overwhelmed with my workload.",
            "chosen": "I understand that feeling of being overwhelmed - it's really common and completely valid. Let's break this down together. What's the most urgent task on your plate right now? Sometimes just identifying the next small step can help reduce that overwhelming feeling. Remember, it's okay to ask for help or to take things one step at a time.",
            "rejected": "Just work harder."
        },
        {
            "prompt": "Can you explain quantum computing in simple terms?",
            "chosen": "Think of quantum computing like having a super-powered calculator that can explore many possibilities at once! While regular computers use bits (like light switches that are either on or off), quantum computers use qubits that can be in multiple states simultaneously. This lets them solve certain problems much faster, like finding the best route through a complex maze or simulating how molecules interact. It's still in early stages, but it's really exciting technology!",
            "rejected": "It's complicated quantum physics stuff."
        }
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample data file: {filename}")


async def example_batch_processing():
    """
    Example 4: Processing multiple items in batch for training
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing for Training")
    print("="*60)
    
    # Use the environment with Hugging Face dataset
    config = ChatbotArenaDPOEnvConfig(
        dataset_name="chatbot_arena_preferences",
        shuffle_dataset=True,
        use_hf_dataset=True,
        max_examples=100  # Small subset for demo
    )
    
    env = ChatbotArenaDPOEnv(
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
            print("\nSample training data:")
            for i, item in enumerate(scored_group.data[:2]):
                print(f"\nTraining Example {i+1}:")
                print(f"  Prompt: {item['prompt'][:100]}...")
                print(f"  Chosen: {item['chosen'][:100]}...")
                print(f"  Rejected: {item['rejected'][:100]}...")
    else:
        print("Failed to process batch")


def example_training_command():
    """
    Example 5: Show how to run the training script
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Training Command Examples")
    print("="*60)
    
    print("To train with Hugging Face dataset (recommended):")
    print("python train_arena_dpo_smol.py \\")
    print("    --use_hf_dataset \\")
    print("    --max_examples 10000 \\")
    print("    --num_train_epochs 3 \\")
    print("    --per_device_train_batch_size 2 \\")
    print("    --learning_rate 5e-5 \\")
    print("    --beta 0.1")
    
    print("\nTo train with your own JSON data:")
    print("python train_arena_dpo_smol.py \\")
    print("    --use_hf_dataset false \\")
    print("    --data_file_path your_arena_data.json \\")
    print("    --num_train_epochs 3 \\")
    print("    --per_device_train_batch_size 2 \\")
    print("    --learning_rate 5e-5")
    
    print("\nTo train with limited data (for testing):")
    print("python train_arena_dpo_smol.py \\")
    print("    --use_hf_dataset \\")
    print("    --max_examples 100 \\")
    print("    --num_train_epochs 1 \\")
    print("    --per_device_train_batch_size 1")
    
    print("\nTo include tie results (not recommended for DPO):")
    print("python train_arena_dpo_smol.py \\")
    print("    --use_hf_dataset \\")
    print("    --filter_ties false \\")
    print("    --max_examples 1000")


def example_data_format():
    """
    Example 6: Show the expected data format
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Data Format Information")
    print("="*60)
    
    print("Hugging Face Dataset Schema:")
    print("- id: Unique identifier")
    print("- model_a: Name of first model")
    print("- model_b: Name of second model")
    print("- prompt: User's input (may be list of strings)")
    print("- response_a: First model's response (may be list of strings)")
    print("- response_b: Second model's response (may be list of strings)")
    print("- winner_model_a: 1 if model A won, 0 otherwise")
    print("- winner_model_b: 1 if model B won, 0 otherwise")
    print("- winner_tie: 1 if it's a tie, 0 otherwise")
    
    print("\nProcessed DPO Format:")
    print("- prompt: User's input message")
    print("- chosen: The preferred/better model response")
    print("- rejected: The non-preferred/worse model response")
    
    print("\nFiltering Options:")
    print("- filter_ties: Remove examples where winner_tie=1")
    print("- filter_both_bad: Remove cases where both responses are bad")
    print("- max_examples: Limit total number of examples")


async def main():
    """
    Run all examples
    """
    print("Chatbot Arena DPO Environment Usage Examples")
    print("="*60)
    
    try:
        # Run examples
        await example_huggingface_dataset()
        await example_basic_environment()
        await example_with_json_data()
        await example_batch_processing()
        example_training_command()
        example_data_format()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\nError: {e}")
        print("This might be due to:")
        print("- No internet connection to download Hugging Face dataset")
        print("- Missing dependencies (run: pip install -r requirements.txt)")
        print("- Dataset access issues")


if __name__ == "__main__":
    asyncio.run(main()) 