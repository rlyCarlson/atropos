#!/usr/bin/env python3
"""
Simple Training Example for Conversational Style DPO

This script shows how to use the conversational style DPO environments
to train a model using the TRL DPOTrainer on Apple Silicon (MPS).

Usage:
    python simple_training_example.py --model_name_or_path distilgpt2 --num_epochs 1
"""

import argparse
import asyncio
import logging
from typing import List

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
import torch

# Import the environments
from conversational_style_dpo_env import (
    ConversationalStyleDPOEnv,
    ConversationalStyleDPOEnvConfig,
)
from gsmk8k_conversational_style_dpo_env import (
    GSM8KConversationalStyleDPOEnv,
    GSM8KConversationalStyleDPOEnvConfig,
)
from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.server_handling.server_baseline import ServerBaseline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_dataset_from_env(env_type: str = "basic", use_api: bool = False):
    """
    Get dataset from the specified environment type.
    
    Args:
        env_type: "basic" or "gsm8k"
        use_api: Whether to use API for dynamic generation (only for gsm8k)
    """
    
    if env_type == "basic":
        # Use the basic environment with synthetic data
        config = ConversationalStyleDPOEnvConfig(
            dataset_name="synthetic_conversational_style",
            shuffle_dataset=True
        )
        
        env = ConversationalStyleDPOEnv(
            config=config,
            server_configs=ServerBaseline(),  # Use ServerBaseline instead of empty list
            testing=True
        )
        
    elif env_type == "gsm8k":
        # Use the GSM8K environment
        if use_api:
            # This would require API key setup
            logger.warning("GSM8K environment with API not implemented in this example")
            logger.info("Falling back to basic environment")
            return await get_dataset_from_env("basic", False)
        else:
            # Use GSM8K environment without API (will use fallback data)
            config = GSM8KConversationalStyleDPOEnvConfig(
                dataset_name="gsm8k_conversational_style",
                shuffle_dataset=True
            )
            
            env = GSM8KConversationalStyleDPOEnv(
                config=config,
                server_configs=ServerBaseline(),  # Use ServerBaseline instead of empty list
                testing=True
            )
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    # Setup the environment
    await env.setup()
    
    # Extract dataset
    if hasattr(env, 'dataset') and env.dataset:
        # For basic environment
        dataset_list = list(env.dataset)
    elif hasattr(env, 'prompt_dataset') and env.prompt_dataset:
        # For GSM8K environment, we need to generate responses
        # This is a simplified version - in practice you'd want to generate responses
        dataset_list = []
        for i in range(min(10, len(env.prompt_dataset))):  # Limit for demo
            item = await env.get_next_item()
            prompt, chosen, rejected = item
            dataset_list.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
    else:
        raise ValueError("No dataset found in environment")
    
    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset_list)
    logger.info(f"Created dataset with {len(hf_dataset)} examples")
    
    return hf_dataset


def main():
    parser = argparse.ArgumentParser(description="Train a model using Conversational Style DPO")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilgpt2",
        help="Model to train (default: distilgpt2)"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        choices=["basic", "gsm8k"],
        default="basic",
        help="Environment type to use (default: basic)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (default: 0.1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dpo_conversational_output",
        help="Output directory for saved model (default: ./dpo_conversational_output)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        help="Maximum prompt length (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon GPU) for training")
    else:
        device = "cpu"
        logger.warning("MPS not available, using CPU for training")
    
    logger.info(f"Starting DPO training with {args.env_type} environment")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Beta: {args.beta}")
    
    # Get dataset
    logger.info("Loading dataset from environment...")
    dataset = asyncio.run(get_dataset_from_env(args.env_type))
    
    if len(dataset) == 0:
        logger.error("No data loaded. Exiting.")
        return
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,  # Use float16 for MPS
        device_map="auto" if device == "mps" else None,  # Auto device mapping for MPS
    )
    
    # Move model to device
    if device == "mps":
        model = model.to(device)
    
    # Set up DPO training arguments
    dpo_training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=50,
        eval_strategy="no",  # No evaluation for this simple example
        save_strategy="epoch",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # MPS-specific optimizations
        bf16=False,
        fp16=False,
        dataloader_num_workers=0 if device == "mps" else 4,  # Reduce workers for MPS
        # DPO-specific arguments
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )
    
    # Initialize DPOTrainer
    logger.info("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will create a copy of the model
        args=dpo_training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        # max_prompt_length=args.max_prompt_length,
        # max_length=args.max_length,
    )
    
    # Start training
    logger.info("Starting training...")
    dpo_trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    dpo_trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 