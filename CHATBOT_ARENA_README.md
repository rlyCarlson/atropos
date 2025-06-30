# Chatbot Arena DPO Training with SmolLM-360M-Instruct

The goal of this project was to train an agent based on the preference pairs from https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k. To accomplish such, I used the conversational_style_dpo example from the community examples. In this example, DPO was used to train distilgpt2 with conversational preference pairs. I replaced the training examples with the arena preference pairs and the model to SmolLM-360M-Instruct. 

When starting this project, my first step was getting this example to work since there was no quick start documentation. After getting that debugged and getting the example code to train (in which I was running into a lot of versioning issues with my environment) I got started on the arena data. 

To train this model run: sh run_chatbot_dpo.sh

Next steps would be to actually finish training the model and have evaluation scripts. One thing I would look out for in this finetuning is model collapse. When I used DPO to finetune SmolLM by juxtaposing similar examples, the model actually collapse, spitting out garbage. In terms of evaluation metrics I would run for this, I would start with an LLMJudge to compare these model outputs with the training data (i.e. using prompts from training data) to sanity check that the model is learning and not collapsing. 