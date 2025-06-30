import asyncio
import logging
import random
import json
import os
from typing import Dict, List, Optional, Tuple

from pydantic import Field
from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChatbotArenaDPOEnvConfig(BaseEnvConfig):
    """Config for ChatbotArenaDPOEnv."""

    dataset_name: str = Field(
        "chatbot_arena_preferences", description="Name of the dataset to use."
    )
    shuffle_dataset: bool = Field(True, description="Whether to shuffle the dataset")
    data_file_path: str = Field(
        "", description="Path to the JSON file containing chatbot arena preference data"
    )
    max_examples: Optional[int] = Field(
        None, description="Maximum number of examples to load (None for all)"
    )
    use_hf_dataset: bool = Field(
        True, description="Whether to use Hugging Face dataset (lmarena-ai/arena-human-preference-55k)"
    )
    filter_ties: bool = Field(
        True, description="Whether to filter out tie results (where winner_tie=1)"
    )
    filter_both_bad: bool = Field(
        True, description="Whether to filter out cases where both responses are bad (winner_tie=1 and both winner_model_a=0, winner_model_b=0)"
    )


class ChatbotArenaDPOEnv(BaseEnv):
    name = "chatbot_arena_dpo"
    name_config_cls = ChatbotArenaDPOEnvConfig

    def __init__(
        self,
        config: ChatbotArenaDPOEnvConfig,
        server_configs: Optional[List[APIServerConfig]] = None,
        slurm=True,
        testing=False,
    ):
        resolved_server_configs = server_configs if server_configs is not None else []
        super().__init__(config, resolved_server_configs, slurm, testing)
        self.config: ChatbotArenaDPOEnvConfig = config
        self.dataset: List[Dict[str, str]] = []
        self.iter: int = 0

    def load_hf_arena_data(self) -> List[Dict[str, str]]:
        """
        Load Chatbot Arena data from Hugging Face dataset.
        Schema: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie
        """
        try:
            logger.info("Loading Chatbot Arena dataset from Hugging Face...")
            dataset = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
            
            processed_data = []
            total_examples = len(dataset)
            logger.info(f"Loaded {total_examples} examples from Hugging Face dataset")
            
            for i, example in enumerate(dataset):
                # Skip ties if configured
                if self.config.filter_ties and example["winner_tie"] == 1:
                    continue
                
                # Skip cases where both responses are bad if configured
                if self.config.filter_both_bad and example["winner_model_a"] == 0 and example["winner_model_b"] == 0:
                    continue
                
                # Extract prompt and responses
                # Handle case where prompt/response might be a list
                prompt = self._extract_text(example["prompt"])
                response_a = self._extract_text(example["response_a"])
                response_b = self._extract_text(example["response_b"])
                
                # Determine which response is preferred
                if example["winner_model_a"] == 1:
                    # Model A won
                    chosen = response_a
                    rejected = response_b
                elif example["winner_model_b"] == 1:
                    # Model B won
                    chosen = response_b
                    rejected = response_a
                else:
                    # It's a tie or both bad, skip
                    continue
                
                # Create DPO format
                processed_item = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "model_a": example["model_a"],
                    "model_b": example["model_b"],
                    "id": example["id"]
                }
                
                processed_data.append(processed_item)
                
                # Log progress
                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i + 1}/{total_examples} examples")
            
            logger.info(f"Successfully processed {len(processed_data)} preference pairs")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Falling back to sample data")
            return self._get_sample_data()

    def _extract_text(self, text_or_list) -> str:
        """
        Extract text from either a string or a list of strings.
        """
        if isinstance(text_or_list, list):
            # Join list elements with newlines
            return "\n".join(str(item) for item in text_or_list)
        else:
            return str(text_or_list)

    def load_arena_data(self) -> List[Dict[str, str]]:
        """
        Load chatbot arena preference data from JSON file or Hugging Face dataset.
        """
        if self.config.use_hf_dataset:
            return self.load_hf_arena_data()
        
        if not self.config.data_file_path:
            logger.warning("No data file path provided. Using sample data.")
            return self._get_sample_data()
        
        if not os.path.exists(self.config.data_file_path):
            logger.error(f"Data file not found: {self.config.data_file_path}")
            logger.warning("Using sample data instead.")
            return self._get_sample_data()
        
        try:
            with open(self.config.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data format
            processed_data = []
            for item in data:
                if isinstance(item, dict) and all(key in item for key in ["prompt", "chosen", "rejected"]):
                    processed_data.append({
                        "prompt": str(item["prompt"]),
                        "chosen": str(item["chosen"]),
                        "rejected": str(item["rejected"])
                    })
                else:
                    logger.warning(f"Skipping invalid data item: {item}")
            
            if self.config.max_examples:
                processed_data = processed_data[:self.config.max_examples]
            
            logger.info(f"Loaded {len(processed_data)} preference pairs from {self.config.data_file_path}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading data from {self.config.data_file_path}: {e}")
            logger.warning("Using sample data instead.")
            return self._get_sample_data()

    def _get_sample_data(self) -> List[Dict[str, str]]:
        """
        Return sample chatbot arena data for testing.
        Replace this with your actual arena data format.
        """
        return [
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
            },
            {
                "prompt": "What should I do if my code isn't working?",
                "chosen": "Don't worry, debugging is a normal part of programming! Let's tackle this step by step. First, what error message are you seeing? If there's no error but unexpected behavior, try adding some print statements to see what's happening at each step. You can also try breaking your code into smaller pieces and testing each part separately. What programming language are you using? I can help you with specific debugging strategies.",
                "rejected": "Check your syntax."
            },
            {
                "prompt": "I want to start a business but don't know where to begin.",
                "chosen": "That's an exciting journey you're about to start! Let's begin with the basics. What problem are you passionate about solving? Start by researching your target market and understanding what people actually need. You don't need a perfect plan to begin - many successful businesses started with just an idea and a willingness to learn. What kind of business are you thinking about? I'd be happy to help you think through the next steps.",
                "rejected": "Just do it."
            }
        ]

    async def setup(self):
        """Load and prepare the chatbot arena dataset."""
        self.dataset = self.load_arena_data()
        
        if self.config.max_examples:
            self.dataset = self.dataset[:self.config.max_examples]
        
        if self.config.shuffle_dataset:
            random.shuffle(self.dataset)
        
        self.iter = 0
        logger.info(f"Loaded chatbot arena dataset with {len(self.dataset)} examples.")

    async def get_next_item(self) -> Item:
        """
        Returns the next item from the dataset.
        For DPO, an "item" will be a tuple of (prompt, chosen_response, rejected_response).
        """
        if not self.dataset or self.iter >= len(self.dataset):
            await self.setup()  # Re-setup if dataset is exhausted or not loaded

        if not self.dataset:  # Still no dataset after setup
            logger.error("Dataset is empty even after setup.")
            # Return a fallback item to avoid crashing
            fallback_prompt = tuple(
                [frozenset({"role": "user", "content": "Fallback prompt"}.items())]
            )
            return (fallback_prompt, "Fallback chosen", "Fallback rejected")

        entry = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1

        # Return the DPO item as (prompt, chosen, rejected)
        return (entry["prompt"], entry["chosen"], entry["rejected"])

    async def collect_trajectories(
        self, items: List[Item]
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Processes DPO items and prepares them for the trainer.
        TRL DPOTrainer will handle the tokenization automatically.
        """
        processed_items = []
        
        for item in items:
            prompt_str, chosen_response_str, rejected_response_str = item
            
            # Handle case where prompt, chosen, and rejected are lists of strings
            if isinstance(prompt_str, list):
                prompt_str = " ".join(prompt_str)
            if isinstance(chosen_response_str, list):
                chosen_response_str = " ".join(chosen_response_str)
            if isinstance(rejected_response_str, list):
                rejected_response_str = " ".join(rejected_response_str)
            
            # Create the data structure that TRL DPOTrainer expects
            processed_item = {
                "prompt": prompt_str,
                "chosen": chosen_response_str,
                "rejected": rejected_response_str,
            }
            processed_items.append(processed_item)
        
        # Create a ScoredDataGroup TypedDict with the processed items
        scored_group: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "advantages": None,
            "ref_logprobs": None,
            "messages": None,
            "group_overrides": None,
            "overrides": None,
            "images": None
        }
        
        # Add the processed items as a custom field for DPO training
        # We'll store the DPO data in the group_overrides field
        scored_group["group_overrides"] = {"dpo_data": processed_items}
        
        return scored_group, items

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        Score the rollout group data.
        For DPO, we don't need to score as we already have preference pairs.
        """
        # This method is not typically used in DPO training
        # as we already have the preference pairs
        return None

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model performance.
        This can be implemented to test the trained model on new conversations.
        """
        logger.info("Evaluation method called - implement custom evaluation logic here")
        return {}

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[ChatbotArenaDPOEnvConfig, List[APIServerConfig]]:
        """
        Initialize the environment configuration.
        """
        config = ChatbotArenaDPOEnvConfig()
        # Create a dummy server config to satisfy BaseEnv requirements
        server_config = APIServerConfig(
            name="dummy",
            base_url="http://localhost:8000",
            api_key="dummy",
            model="dummy"
        )
        server_configs = [server_config]
        return config, server_configs

    @classmethod
    async def main_test(cls):
        """
        Test the environment setup and data loading.
        """
        config, server_configs = cls.config_init()
        env = cls(config=config, server_configs=server_configs, testing=True)
        
        await env.setup()
        
        # Test getting a few items
        for i in range(3):
            item = await env.get_next_item()
            prompt, chosen, rejected = item
            print(f"\nTest Item {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Chosen: {chosen}")
            print(f"Rejected: {rejected}")
        
        # Test trajectory collection
        items = []
        for _ in range(2):
            item = await env.get_next_item()
            items.append(item)
        
        scored_group, processed_items = await env.collect_trajectories(items)
        
        if scored_group and scored_group.get("group_overrides", {}).get("dpo_data"):
            dpo_data = scored_group["group_overrides"]["dpo_data"]
            print(f"\nSuccessfully created ScoredDataGroup with {len(dpo_data)} items")
            print(f"Sample processed item: {dpo_data[0]}")
        else:
            print("Failed to create ScoredDataGroup")


if __name__ == "__main__":
    # Test the environment
    asyncio.run(ChatbotArenaDPOEnv.main_test()) 