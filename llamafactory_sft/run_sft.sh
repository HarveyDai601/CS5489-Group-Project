#!/usr/bin/env bash
set -euo pipefail

# Allow callers to override paths without editing the script.
CONFIG_FILE=${CONFIG_FILE:-configs/lora_mt.yaml}
DATA_DIR=${DATA_DIR:-llamafactory_sft/data}
DATASET_STEM=${DATASET_STEM:-wmt14_en_de}
LLAMA_CONFIG=${LLAMA_CONFIG:-llamafactory_sft/configs/mbart_wmt14_en_de.yaml}

# 1) Regenerate JSONL data in the conversation format expected by LlamaFactory.
python llamafactory_sft/prepare_wmt14_sft.py \
  --config "$CONFIG_FILE" \
  --output_dir "$DATA_DIR" \
  --dataset_stem "$DATASET_STEM"

# 2) Launch LoRA SFT using the official CLI (assumes `llamafactory-cli` is installed).
llamafactory-cli train "$LLAMA_CONFIG"
