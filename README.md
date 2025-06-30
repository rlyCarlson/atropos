# Conversational Style DPO Environments

This directory contains environments for training language models using Direct Preference Optimization (DPO) to prefer more engaging, empathetic, and helpful conversational responses over blunt or unhelpful ones.

## Overview

The conversational style DPO environments are designed to train models to generate responses that are:
- **Engaging**: Encourage further conversation
- **Empathetic**: Show understanding and care
- **Helpful**: Provide useful information and guidance
- **Clear**: Easy to understand and follow

## Environments

### 1. ConversationalStyleDPOEnv (Basic)

A simple environment that uses predefined synthetic data with prompt/response pairs. This environment doesn't require any API calls and is perfect for testing and development.

**Features:**
- Pre-defined prompt/response pairs
- No external API dependencies
- Fast setup and execution
- Good for prototyping and testing

### 2. GSM8KConversationalStyleDPOEnv (Advanced)

A more sophisticated environment that dynamically generates prompts and responses using an LLM API. This environment can create diverse, high-quality training data on-the-fly.

**Features:**
- Dynamic prompt generation via LLM
- Real-time response generation
- Configurable generation parameters
- Higher quality and more diverse data

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For the GSM8K environment, set up your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Usage

Run the usage examples to see how the environments work:

```bash
python usage_example.py
```

This will demonstrate:
- Basic environment setup and data loading
- GSM8K environment with dynamic generation
- Batch processing for training
- Configuration options
- Custom dataset creation

### Training a Model

Use the simple training script to train a model with DPO:

```bash
# Basic training with default settings
python simple_training_example.py

# Custom training parameters
python simple_training_example.py \
    --model_name_or_path distilgpt2 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --beta 0.1
```

### Using the Existing Training Script

The original training script provides more advanced features:

```bash
python train_dpo_conversational.py \
    --model_name_or_path distilgpt2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --beta 0.1
```

## Environment Configuration

### ConversationalStyleDPOEnvConfig

```python
config = ConversationalStyleDPOEnvConfig(
    dataset_name="synthetic_conversational_style",
    shuffle_dataset=True
)
```

**Parameters:**
- `dataset_name`: Name of the dataset (default: "synthetic_conversational_style")
- `shuffle_dataset`: Whether to shuffle the dataset (default: True)

### GSM8KConversationalStyleDPOEnvConfig

```python
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
```

**Parameters:**
- `dataset_name`: Name of the dataset
- `shuffle_dataset`: Whether to shuffle the dataset
- `chosen_temperature`: Temperature for generating preferred responses
- `chosen_max_tokens`: Maximum tokens for preferred responses
- `rejected_temperature`: Temperature for generating non-preferred responses
- `rejected_max_tokens`: Maximum tokens for non-preferred responses
- `prompt_generation_temperature`: Temperature for generating prompts
- `prompt_generation_max_tokens`: Maximum tokens for prompt generation
- `data_path_to_save_groups`: Path to save processed data

## Data Format

Both environments produce data in the format expected by TRL's DPOTrainer:

```python
{
    "prompt": "User's input message",
    "chosen": "Preferred response (engaging, helpful, empathetic)",
    "rejected": "Non-preferred response (blunt, unhelpful, generic)"
}
```

## Example Data

### Preferred Responses (Chosen)
- "I'm sorry to hear that. Sometimes a little self-care can help. What's one small thing you could do for yourself right now?"
- "That's fantastic news! Tell me more about it - what are you most looking forward to?"
- "I can understand that some concepts can be tricky! Could you tell me which part is confusing you? Maybe we can break it down together."

### Non-Preferred Responses (Rejected)
- "Okay."
- "Good for you."
- "Read the manual."

## Customization

### Creating Custom Data

You can create your own dataset by subclassing the basic environment:

```python
class CustomConversationalStyleDPOEnv(ConversationalStyleDPOEnv):
    async def setup(self):
        self.synthetic_data = [
            {
                "prompt": "Your custom prompt",
                "chosen": "Your preferred response",
                "rejected": "Your non-preferred response"
            }
            # ... more examples
        ]
        # ... rest of setup
```

### Modifying Generation Parameters

For the GSM8K environment, you can adjust the generation parameters to control response quality:

```python
config = GSM8KConversationalStyleDPOEnvConfig(
    chosen_temperature=0.8,  # More creative chosen responses
    rejected_temperature=0.2,  # More deterministic rejected responses
    chosen_max_tokens=200,  # Longer chosen responses
    rejected_max_tokens=30   # Shorter rejected responses
)
```

## Training Tips

1. **Start Small**: Begin with a small model like `distilgpt2` for testing
2. **Use Small Datasets**: Start with a few epochs and small batch sizes
3. **Monitor Loss**: Watch the DPO loss to ensure training is progressing
4. **Validate Results**: Test the trained model on conversational tasks
5. **Iterate**: Adjust parameters based on results

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed all dependencies from `requirements.txt`
2. **API Key Issues**: For GSM8K environment, ensure your API key is set correctly
3. **Memory Issues**: Reduce batch size or use a smaller model
4. **Tokenization Errors**: Ensure your tokenizer supports the model you're using

### Getting Help

- Check the usage examples in `usage_example.py`
- Review the training script in `train_dpo_conversational.py`
- Look at the environment source code for implementation details

## Advanced Usage

### Integration with Atropos Framework

These environments are designed to work with the Atropos framework. You can use them with other Atropos components for more advanced training pipelines.

### Batch Processing

For efficient training, the environments support batch processing:

```python
items = []
for _ in range(batch_size):
    item = await env.get_next_item()
    items.append(item)

scored_group, processed_items = await env.collect_trajectories(items)
```

### Evaluation

You can evaluate trained models by comparing their responses to the preferred responses in your dataset or by using human evaluation.

## Contributing

To contribute to these environments:

1. Add new prompt/response pairs to the synthetic data
2. Improve the generation prompts for the GSM8K environment
3. Add new configuration options
4. Create additional environment variants
5. Improve documentation and examples

## License

This code is part of the Atropos project. Please refer to the main project license for details. 