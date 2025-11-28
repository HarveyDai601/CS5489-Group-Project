# LlamaFactory SFT pipeline

This folder contains everything needed to turn the existing WMT14 EN→DE data into the conversation-style JSONL format expected by [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) and to launch a supervised fine-tuning (SFT) run with LoRA.

## 1. Prepare JSONL data

```bash
python llamafactory_sft/prepare_wmt14_sft.py \
  --config configs/lora_mt.yaml \
  --output_dir llamafactory_sft/data \
  --dataset_stem wmt14_en_de
```

Key points:
- Uses the same YAML config as `train_lora_cuda.py` to stay in sync with dataset names, prompts, and sample limits.
- Emits `wmt14_en_de_train.jsonl` and `wmt14_en_de_validation.jsonl` files stored under `llamafactory_sft/data/`.
- Each record follows LlamaFactory's `conversation` schema (`user` prompt, `assistant` translation) so it can be consumed directly.

## 2. Run LlamaFactory SFT

Adapt the provided config (`llamafactory_sft/configs/mbart_wmt14_en_de.yaml`) as needed, then launch training via the helper script:

```bash
bash llamafactory_sft/run_sft.sh
```

What the script does:
1. Regenerates the JSONL files (safe to rerun; files are overwritten).
2. Calls `llamafactory-cli train` with the same hyper-parameters used in the CUDA LoRA flow, targeting the new dataset files.

Make sure `llamafactory-cli` is available in your environment (installable via `pip install llamafactory`). Set `CUDA_VISIBLE_DEVICES`, `WANDB_PROJECT`, etc., before running the script if needed.
