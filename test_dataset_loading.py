#!/usr/bin/env python3
"""
Simple test script to debug Hugging Face dataset loading
"""

import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hf_dataset_loading():
    """Test loading the Chatbot Arena dataset from Hugging Face"""
    try:
        logger.info("Loading Chatbot Arena dataset from Hugging Face...")
        dataset = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
        
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
        
        # Check the first example
        first_example = dataset[0]
        logger.info(f"First example keys: {list(first_example.keys())}")
        logger.info(f"First example: {first_example}")
        
        # Test processing a few examples
        processed_data = []
        for i, example in enumerate(dataset):
            if i >= 5:  # Only process first 5 examples
                break
                
            # Skip ties
            if example["winner_tie"] == 1:
                continue
            
            # Skip cases where both responses are bad
            if example["winner_tie"] == 1 and example["winner_model_a"] == 0 and example["winner_model_b"] == 0:
                continue
            
            # Extract prompt and responses
            prompt = extract_text(example["prompt"])
            response_a = extract_text(example["response_a"])
            response_b = extract_text(example["response_b"])
            
            # Determine which response is preferred
            if example["winner_model_a"] == 1:
                chosen = response_a
                rejected = response_b
            elif example["winner_model_b"] == 1:
                chosen = response_b
                rejected = response_a
            else:
                continue
            
            processed_item = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "model_a": example["model_a"],
                "model_b": example["model_b"],
                "id": example["id"]
            }
            
            processed_data.append(processed_item)
            logger.info(f"Processed example {i+1}: {processed_item}")
        
        logger.info(f"Successfully processed {len(processed_data)} preference pairs")
        return True
        
    except Exception as e:
        logger.error(f"Error loading Hugging Face dataset: {e}")
        return False

def extract_text(text_or_list):
    """Extract text from either a string or a list of strings."""
    if isinstance(text_or_list, list):
        return "\n".join(str(item) for item in text_or_list)
    else:
        return str(text_or_list)

if __name__ == "__main__":
    success = test_hf_dataset_loading()
    if success:
        print("✅ Dataset loading test successful!")
    else:
        print("❌ Dataset loading test failed!") 