"""
Generate CSV Report (Final V2 - Conservative Mode)
策略回退：
1. 去掉 decoder_start_token_id (这导致了崩塌)
2. 将 Scaling 进一步降低到 0.1 (极致求稳)
3. 保留 bfloat16
"""
import argparse
import csv
import logging
import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset

import warnings
import sacrebleu

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger(__name__)

# === 本地路径 ===
LOCAL_MODEL_PATH = "/root/CS5489-Group-Project/cache/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/e30b6cb8eb0d43a0b73cab73c7676b9863223a30"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--base_model", type=str, default=LOCAL_MODEL_PATH)
    return parser.parse_args()

def load_yaml_config(path):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_yaml_config(args.config)
    data_cfg = cfg.get("data", {})
    src_lang = "en_XX"
    tgt_lang = "de_DE"

    # 1. 加载模型 (bfloat16)
    LOGGER.info(f"⏳ 加载模型 [bfloat16]...")
    model = MBartForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )

    LOGGER.info(f"⏳ 加载 LoRA: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    # 2. 极致调整 Scaling -> 0.1
    # 既然 0.5 还是崩，我们压到 0.1，宁可翻译得不准，也不能乱码
    TARGET_SCALE = 0.3
    LOGGER.info(f"🔧 [策略调整] 强制 Scaling = {TARGET_SCALE}")
    for name, module in model.named_modules():
        if "lora" in str(type(module)).lower() and hasattr(module, "scaling"):
            module.scaling = {"default": TARGET_SCALE}

    # 3. Tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.src_lang = src_lang
    tgt_lang_id = tokenizer.lang_code_to_id[tgt_lang]

    # 4. 数据
    dataset_name = data_cfg.get("dataset_name", "wmt14")
    dataset_subset = data_cfg.get("dataset_config", "de-en")
    LOGGER.info(f"📚 加载数据 {dataset_name}/{dataset_subset}...")
    try:
        dataset = load_dataset(dataset_name, dataset_subset, split=args.split, streaming=True)
    except:
        dataset = load_dataset(dataset_name, dataset_subset, split="validation", streaming=True)
    samples = list(dataset.take(args.num_samples))

    # 5. 推理
    results = []
    bleu_metric = BLEU()
    text_col = data_cfg.get("text_column", "translation")

    for i in tqdm(range(0, len(samples), args.batch_size), desc="Generating"):
        batch_samples = samples[i : i + args.batch_size]
        src_texts = [s[text_col]['en'] for s in batch_samples]
        ref_texts = [s[text_col]['de'] for s in batch_samples]

        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                # decoder_start_token_id=tgt_lang_id, # <--- 删掉了这个！！
                max_length=128,
                num_beams=5,
                repetition_penalty=1.2, # 降低一点惩罚，防止为了不重复而乱说话
                no_repeat_ngram_size=3
            )

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for src, ref, pred in zip(src_texts, ref_texts, decoded_preds):
            score_obj = bleu_metric.sentence_score(pred, [ref])
            results.append({
                "input": src,
                "output": pred,
                "label": ref,
                "bleu": round(score_obj.score, 4)
            })

    # 6. 保存
    LOGGER.info(f"💾 保存至 {args.csv_path}")
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output", "label", "bleu"])
        writer.writeheader()
        writer.writerows(results)

    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    LOGGER.info(f"📊 平均 BLEU: {avg_bleu:.2f}")

if __name__ == "__main__":
    main()
