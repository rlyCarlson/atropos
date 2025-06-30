#!/usr/bin/env python3
"""
Test Hugging Face dataset loading directly
"""

from datasets import load_dataset

def test_hf_dataset():
    """Test loading the Hugging Face dataset directly"""
    try:
        print("Loading dataset...")
        dataset = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
        print(f"Dataset loaded successfully with {len(dataset)} examples")
        
        if len(dataset) > 0:
            print(f"First example: {dataset[0]}")
            print(f"Columns: {dataset.column_names}")
            
            # Test accessing a few examples
            for i in range(min(5, len(dataset))):
                example = dataset[i]
                print(f"\nExample {i}:")
                print(f"  ID: {example.get('id', 'N/A')}")
                print(f"  Prompt: {example.get('prompt', 'N/A')}")
                print(f"  Response A: {example.get('response_a', 'N/A')}")
                print(f"  Response B: {example.get('response_b', 'N/A')}")
                print(f"  Winner A: {example.get('winner_model_a', 'N/A')}")
                print(f"  Winner B: {example.get('winner_model_b', 'N/A')}")
                print(f"  Tie: {example.get('winner_tie', 'N/A')}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hf_dataset() 