# Chatbot Arena DPO Training with SmolLM-360M-Instruct

This directory contains a custom environment and training script for finetuning SmolLM-360M-Instruct using Direct Preference Optimization (DPO) on chatbot arena preference data.

## Overview

The Chatbot Arena DPO environment allows you to:
- Load preference pairs from the real Chatbot Arena dataset ([lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k))
- Train models to prefer better responses over worse ones
- Use SmolLM-360M-Instruct as the base model
- Integrate with the Atropos framework

## Files

- `chatbot_arena_dpo_env.py` - Custom environment for loading arena preference data
- `train_arena_dpo_smol.py` - Training script for SmolLM-360M-Instruct
- `sample_arena_data.json` - Example data format
- `arena_dpo_usage_example.py` - Usage examples and demonstrations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with Hugging Face Dataset (Recommended)

```bash
python train_arena_dpo_smol.py \
    --use_hf_dataset \
    --max_examples 10000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --beta 0.1
```

### 3. Train with Your Own Data

Convert your chatbot arena results to this JSON format:

```json
[
  {
    "prompt": "User's input message",
    "chosen": "The preferred/better model response",
    "rejected": "The non-preferred/worse model response"
  },
  ...
]
```

Then run:

```bash
python train_arena_dpo_smol.py \
    --use_hf_dataset false \
    --data_file_path your_arena_data.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5
```

### 4. Test the Environment

```bash
python arena_dpo_usage_example.py
```

## Data Sources

### Hugging Face Dataset (Recommended)

The environment can automatically load the real Chatbot Arena dataset from Hugging Face:
- **Dataset**: [lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
- **Size**: ~57,000 preference pairs
- **Models**: 70+ state-of-the-art LLMs including GPT-4, Claude 2, Llama 2, Gemini, Mistral
- **Format**: Real human preferences from Chatbot Arena battles

### Custom JSON Data

You can also provide your own preference data in JSON format with these fields:
- **prompt**: The user's input message or question
- **chosen**: The response that was preferred in your arena
- **rejected**: The response that was not preferred

## Environment Configuration

The `ChatbotArenaDPOEnvConfig` supports these parameters:

- `dataset_name`: Name of the dataset (default: "chatbot_arena_preferences")
- `shuffle_dataset`: Whether to shuffle the dataset (default: True)
- `use_hf_dataset`: Whether to use Hugging Face dataset (default: True)
- `data_file_path`: Path to your JSON data file (used when use_hf_dataset=False)
- `max_examples`: Maximum number of examples to load (None for all)
- `filter_ties`: Whether to filter out tie results (default: True)
- `filter_both_bad`: Whether to filter out cases where both responses are bad (default: True)

## Training Configuration

The training script supports these key parameters:

### Model Parameters
- `--model_name_or_path`: Model to finetune (set to SmolLM-360M-Instruct)
- `--beta`: DPO beta parameter (default: 0.1)
- `--max_prompt_length`: Max length for prompts (default: 512)
- `--max_length`: Max length for responses including prompt (default: 1024)

### Data Parameters
- `--use_hf_dataset`: Use Hugging Face dataset (default: True)
- `--data_file_path`: Path to JSON data file (when use_hf_dataset=False)
- `--max_examples`: Limit number of examples for training
- `--filter_ties`: Filter out tie results (default: True)
- `--filter_both_bad`: Filter out cases where both responses are bad (default: True)

### Training Parameters
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--learning_rate`: Learning rate

## Usage Examples

### Training with Hugging Face Dataset

```bash
# Full dataset training
python train_arena_dpo_smol.py \
    --use_hf_dataset \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5

# Limited data for testing
python train_arena_dpo_smol.py \
    --use_hf_dataset \
    --max_examples 1000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1
```

### Training with Custom Data

```bash
python train_arena_dpo_smol.py \
    --use_hf_dataset false \
    --data_file_path your_arena_data.json \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4
```

### Advanced Training Options

```bash
python train_arena_dpo_smol.py \
    --use_hf_dataset \
    --max_examples 5000 \
    --filter_ties false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --beta 0.2 \
    --max_prompt_length 256 \
    --max_length 512
```

## Dataset Processing

### Hugging Face Dataset Schema

The original dataset has this schema:
- `id`: Unique identifier
- `model_a`: Name of first model
- `model_b`: Name of second model
- `prompt`: User's input (may be list of strings)
- `response_a`: First model's response (may be list of strings)
- `response_b`: Second model's response (may be list of strings)
- `winner_model_a`: 1 if model A won, 0 otherwise
- `winner_model_b`: 1 if model B won, 0 otherwise
- `winner_tie`: 1 if it's a tie, 0 otherwise

### Processing Logic

The environment processes the data as follows:
1. **Loads** the Hugging Face dataset
2. **Filters** out ties and both-bad cases (configurable)
3. **Determines** which response is preferred based on winner flags
4. **Converts** to DPO format: `{"prompt", "chosen", "rejected"}`
5. **Handles** text that may be in list format by joining with newlines

## Environment Features

### Automatic Data Loading
The environment automatically downloads and processes the Hugging Face dataset.

### Smart Filtering
- Filters out tie results (where neither response is preferred)
- Filters out cases where both responses are considered bad
- Configurable filtering options

### Fallback Options
- Falls back to sample data if Hugging Face dataset is unavailable
- Supports local JSON files as alternative data source

### Batch Processing
The environment supports batch processing for efficient training.

### Integration with TRL
The environment produces data in the exact format expected by TRL's DPOTrainer.

## SmolLM-360M-Instruct Specific Configuration

The training script is optimized for SmolLM-360M-Instruct with:

- Automatic device mapping
- Mixed precision training (fp16)
- Proper tokenizer configuration
- Special token handling
- Memory-efficient settings

## Troubleshooting

### Common Issues

1. **Dataset Download Issues**
   - Ensure you have internet access
   - Check Hugging Face dataset availability
   - Try using local JSON data as fallback

2. **Memory Issues**
   - Reduce batch size: `--per_device_train_batch_size 1`
   - Reduce sequence length: `--max_length 512`
   - Limit examples: `--max_examples 1000`
   - Use gradient accumulation: `--gradient_accumulation_steps 4`

3. **Model Loading Issues**
   - Ensure you have the correct model path
   - Check that the model supports causal language modeling
   - Verify you have sufficient disk space

4. **Training Issues**
   - Start with a small dataset for testing
   - Use a lower learning rate: `--learning_rate 1e-5`
   - Monitor loss curves for convergence

### Getting Help

- Run the usage examples: `python arena_dpo_usage_example.py`
- Check the sample data format: `sample_arena_data.json`
- Review the environment source code for implementation details

## Advanced Usage

### Custom Environment Modifications

You can modify the environment to:
- Add data preprocessing
- Implement custom validation
- Add data augmentation
- Support different data formats

### Integration with Atropos

The environment integrates with the Atropos framework:
- Uses Atropos base classes
- Supports Atropos configuration patterns
- Compatible with Atropos training pipelines

### Evaluation

After training, you can:
- Test the model on new conversations
- Compare responses to your preference data
- Use human evaluation
- Implement automated metrics

## Performance Tips

1. **Data Quality**: The Hugging Face dataset contains high-quality human preferences
2. **Batch Size**: Start small and increase based on your GPU memory
3. **Learning Rate**: Use 5e-5 to 1e-4 for most cases
4. **Beta Parameter**: 0.1 is a good starting point, adjust based on preference strength
5. **Sequence Length**: Balance between context and memory usage
6. **Filtering**: Use filtering to focus on clear preference pairs

## Expected Results

With the real Chatbot Arena data, you should see:
- Improved response quality based on human preferences
- Better alignment with preferred conversational styles
- Reduced generation of rejected response patterns
- More engaging and helpful responses

## Citation

If you use the Chatbot Arena dataset, please cite:

```bibtex
@misc{chiang2024chatbot,
    title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
    author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
    year={2024},
    eprint={2403.04132},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Contributing

To improve this environment:
1. Add support for more data formats
2. Implement additional evaluation metrics
3. Add data augmentation techniques
4. Optimize for different model architectures
5. Add support for multi-turn conversations

## License

This code is part of the Atropos project. Please refer to the main project license for details. 