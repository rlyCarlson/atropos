#!/usr/bin/env python3
"""
Chatbot Arena DPO Training Script for SmolLM-360M-Instruct

This script finetunes SmolLM-360M-Instruct using Direct Preference Optimization (DPO)
on chatbot arena preference data from the Hugging Face dataset.

Usage:
    python train_arena_dpo_smol.py --data_file_path your_arena_data.json

Requirements:
    - Install dependencies: pip install -r requirements.txt
    - Prepare your arena data in JSON format (or use Hugging Face dataset)
    - Ensure you have sufficient GPU memory for SmolLM-360M-Instruct
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Add the atropos root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig
from atroposlib.envs.base import APIServerConfig

# Import our custom environment
from chatbot_arena_dpo_env import (
    ChatbotArenaDPOEnv,
    ChatbotArenaDPOEnvConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the Chatbot Arena DPO training script.
    """

    model_name_or_path: str = field(
        default="microsoft/DialoGPT-medium",  # Will be overridden to SmolLM
        metadata={"help": "The model name or path to load from."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer name or path. Defaults to model_name_or_path."
        },
    )
    data_file_path: str = field(
        default="",
        metadata={"help": "Path to JSON file containing chatbot arena preference data"}
    )
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of examples to use for training"}
    )
    use_hf_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to use Hugging Face dataset (lmarena-ai/arena-human-preference-55k)"}
    )
    filter_ties: bool = field(
        default=True,
        metadata={"help": "Whether to filter out tie results"}
    )
    filter_both_bad: bool = field(
        default=True,
        metadata={"help": "Whether to filter out cases where both responses are bad"}
    )
    output_dir: str = field(default="./smol_arena_dpo_results")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)


async def get_dataset_from_env(
    env_config: ChatbotArenaDPOEnvConfig,
) -> Dataset:
    """
    Initializes the environment and extracts the chatbot arena dataset
    in the format required by DPOTrainer.
    """
    dummy_server_config = APIServerConfig(
        name="dummy",
        base_url="http://localhost:8000",
        api_key="dummy",
        model="dummy"
    )
    env = ChatbotArenaDPOEnv(config=env_config, server_configs=[dummy_server_config], testing=True)
    await env.setup()

    if not env.dataset:
        raise ValueError(
            "Dataset is empty after environment setup. Check your data file path or Hugging Face dataset access."
        )

    logger.info(f"Environment dataset has {len(env.dataset)} items")
    
    # Debug: Check the first few items
    if len(env.dataset) > 0:
        logger.info(f"First dataset item: {env.dataset[0]}")
        logger.info(f"Dataset item keys: {list(env.dataset[0].keys()) if isinstance(env.dataset[0], dict) else 'Not a dict'}")

    # Ensure all items have the required format
    processed_dataset = []
    for i, item in enumerate(env.dataset):
        try:
            if isinstance(item, dict) and all(key in item for key in ["prompt", "chosen", "rejected"]):
                # Handle list format for prompt, chosen, rejected
                prompt = item["prompt"]
                chosen = item["chosen"]
                rejected = item["rejected"]
                
                if isinstance(prompt, list):
                    prompt = " ".join(prompt)
                if isinstance(chosen, list):
                    chosen = " ".join(chosen)
                if isinstance(rejected, list):
                    rejected = " ".join(rejected)
                
                processed_item = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
                processed_dataset.append(processed_item)
            else:
                logger.warning(f"Skipping malformed item {i}: {item}")
        except Exception as e:
            logger.warning(f"Error processing item {i}: {e}")
            continue

    if not processed_dataset:
        raise ValueError("No valid items found in dataset after processing")

    logger.info(f"Processed {len(processed_dataset)} valid items")

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(processed_dataset)

    # Log a sample to verify
    if len(hf_dataset) > 0:
        logger.info(f"Sample from dataset: {hf_dataset[0]}")
        logger.info(f"Dataset columns: {hf_dataset.column_names}")
    else:
        logger.warning("Dataset created from environment is empty!")

    return hf_dataset


def setup_smol_model_and_tokenizer(model_name_or_path: str, tokenizer_name_or_path: str, device: str):
    """
    Set up SmolLM-360M-Instruct model and tokenizer with proper configuration.
    """
    logger.info(f"Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    # Ensure tokenizer has necessary tokens
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add special tokens if needed for instruction following
    # Note: SmolLM might already have these tokens, so we'll check first
    existing_tokens = set(tokenizer.get_vocab().keys())
    special_tokens_to_add = []
    
    for token in ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>"]:
        if token not in existing_tokens:
            special_tokens_to_add.append(token)
    
    if special_tokens_to_add:
        logger.info(f"Adding special tokens: {special_tokens_to_add}")
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    
    logger.info(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,  # Explicitly use float32
        trust_remote_code=True,  # Required for some custom models
    )
    
    # Move model to device explicitly
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Resize token embeddings if we added special tokens
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info("Resizing token embeddings to match tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def main():
    parser = HfArgumentParser((ScriptArguments,))
    script_args = parser.parse_args_into_dataclasses()[0]

    # Check MPS availability
    if torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon GPU) is available!")
        device = "mps"
    else:
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"

    if script_args.tokenizer_name_or_path is None:
        script_args.tokenizer_name_or_path = script_args.model_name_or_path

    # Set SmolLM-360M-Instruct model path
    script_args.model_name_or_path = "HuggingFaceTB/SmolLM2-135M-Instruct"
    script_args.tokenizer_name_or_path = script_args.model_name_or_path
    
    logger.info(f"Using model: {script_args.model_name_or_path}")
    logger.info(f"Using tokenizer: {script_args.tokenizer_name_or_path}")

    # --- 1. Initialize Environment and Get Dataset ---
    logger.info("Initializing Chatbot Arena DPO environment...")
    
    # Create environment config with data file path
    env_config = ChatbotArenaDPOEnvConfig(
        dataset_name="chatbot_arena_preferences",
        shuffle_dataset=True,
        data_file_path=script_args.data_file_path,
        max_examples=script_args.max_examples,
        use_hf_dataset=script_args.use_hf_dataset,
        filter_ties=script_args.filter_ties,
        filter_both_bad=script_args.filter_both_bad
    )
    
    # Set tokenizer name for the environment
    env_config.tokenizer_name = script_args.tokenizer_name_or_path

    # Get dataset from environment
    try:
        dataset = asyncio.run(get_dataset_from_env(env_config))
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure you have internet access to download the Hugging Face dataset")
        logger.error("Or provide a valid data_file_path for local JSON data")
        return

    if len(dataset) == 0:
        logger.error("No data loaded. Exiting training.")
        return

    # --- 2. Load Model and Tokenizer ---
    try:
        model, tokenizer = setup_smol_model_and_tokenizer(
            script_args.model_name_or_path, 
            script_args.tokenizer_name_or_path,
            device
        )
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        return

    # Reference model for DPO (will be a copy of the policy model)
    model_ref = None
    logger.info("Reference model will be a copy of the policy model (handled by DPOTrainer).")

    # --- 3. Create DPOConfig ---
    dpo_config = DPOConfig(
        output_dir="./smol_arena_dpo_results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        # Precision settings for MPS (Apple Silicon)
        fp16=False,  # MPS doesn't support fp16 training
        bf16=False,  # MPS doesn't support bf16
        dataloader_pin_memory=False,  # Disable for MPS
        remove_unused_columns=False,
        report_to="none",  # Set to "wandb" if you want logging
        # DPO-specific parameters
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )

    logger.info(f"DPO Config: {dpo_config}")

    # --- 4. Initialize DPOTrainer ---
    logger.info("Initializing DPOTrainer...")
    logger.info(f"Using device: {device}")
    logger.info(f"DPO Config precision settings: fp16={dpo_config.fp16}, bf16={dpo_config.bf16}")
    
    try:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=None,  # Set to PEFT config if using LoRA
        )
        logger.info("DPOTrainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DPOTrainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 5. Start Training ---
    logger.info("Starting DPO training...")
    try:
        dpo_trainer.train()
        logger.info("DPO training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # --- 6. Save the Model ---
    if dpo_config.should_save:
        output_save_dir = dpo_config.output_dir
        logger.info(f"Saving model to {output_save_dir}")
        try:
            dpo_trainer.save_model(output_save_dir)
            tokenizer.save_pretrained(output_save_dir)
            logger.info("Model and tokenizer saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    # --- 7. Optional: Test the trained model ---
    logger.info("Training complete! You can now test your finetuned model.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main() 