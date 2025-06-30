#!/bin/bash

python train_arena_dpo_smol.py \
    --output_dir ./smol_arena_dpo_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --max_examples 1000 \
    --use_hf_dataset True