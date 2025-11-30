"""Generate CSV examples from a LoRA fine-tuned translation model."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from peft import PeftModel
from sacrebleu.metrics import BLEU
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

from data_utils import load_translation_dataset

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV report examples from a fine-tuned model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lora_mt.yaml"),
        help="Path to the YAML config file used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory that contains the saved LoRA adapter weights (e.g. outputs/...)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to run inference on (train/validation/test).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="Number of samples to include in the CSV output.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation to balance throughput and memory use.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("outputs/report_examples.csv"),
        help="Destination CSV file path.",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_dtype(name: str | None) -> torch.dtype | None:
    if not name:
        return None
    lowered = name.lower()
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float16", "fp16"}:
        return torch.float16
    return torch.float32


def batched(iterable: List[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def build_records(dataset, data_cfg: Dict[str, Any], limit: int, split_name: str) -> List[Dict[str, str]]:
    source_language = data_cfg["source_language"]
    target_language = data_cfg["target_language"]
    text_column = data_cfg["text_column"]
    prompt_template = data_cfg["prompt_template"]

    total = min(limit, len(dataset))
    LOGGER.info("Sampling %d examples from %s split", total, split_name)

    records: List[Dict[str, str]] = []
    for example in dataset.select(range(total)):
        translation = example[text_column]
        source_text = translation[source_language]
        target_text = translation[target_language]
        prompt = prompt_template.format(
            source_text=source_text,
            target_text=target_text,
            source_lang=source_language,
            target_lang=target_language,
        )
        records.append({"input": prompt, "label": target_text})
    return records


def build_generation_kwargs(training_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "generation_max_length": "max_length",
        "generation_min_length": "min_length",
        "generation_num_beams": "num_beams",
        "generation_do_sample": "do_sample",
        "generation_temperature": "temperature",
        "generation_top_k": "top_k",
        "generation_top_p": "top_p",
        "generation_no_repeat_ngram_size": "no_repeat_ngram_size",
    }
    kwargs: Dict[str, Any] = {}
    for cfg_key, gen_key in mapping.items():
        value = training_cfg.get(cfg_key)
        if value is not None:
            kwargs[gen_key] = value
    kwargs.setdefault("max_length", data_cfg.get("max_target_length", 256))
    kwargs.setdefault("num_beams", 4)
    return kwargs


def generate_outputs(
    model,
    tokenizer,
    records: List[Dict[str, str]],
    device: torch.device,
    batch_size: int,
    data_cfg: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
) -> None:
    max_source_length = data_cfg.get("max_source_length", 512)
    model.eval()
    with torch.no_grad():
        for batch in batched(records, batch_size):
            texts = [record["input"] for record in batch]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            ).to(device)
            outputs = model.generate(**inputs, **gen_kwargs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for record, prediction in zip(batch, decoded):
                record["output"] = prediction.strip()


def write_csv(records: List[Dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input", "output", "label", "bleu"])
        writer.writeheader()
        writer.writerows(records)


def annotate_bleu_scores(records: List[Dict[str, str]]) -> float:
    if not records:
        return 0.0
    bleu_metric = BLEU()
    predictions: List[str] = []
    references: List[str] = []
    for record in records:
        prediction = record.get("output", "").strip()
        reference = record.get("label", "").strip()
        if prediction and reference:
            sentence_score = bleu_metric.sentence_score(prediction, [reference])
            record["bleu"] = round(sentence_score.score, 4)
        else:
            record["bleu"] = ""
        predictions.append(prediction)
        references.append(reference)

    corpus_score = bleu_metric.corpus_score(predictions, [references])
    return corpus_score.score


def main():
    args = parse_args()
    configure_logging()

    config = load_config(args.config)
    project_cfg = config["project"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    tokenizer = MBart50TokenizerFast.from_pretrained(model_cfg["name"], cache_dir=project_cfg.get("cache_dir"))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.src_lang = model_cfg["tokenizer_src_lang_code"]
    tokenizer.tgt_lang = model_cfg["tokenizer_tgt_lang_code"]

    target_lang_code = model_cfg["tokenizer_tgt_lang_code"]
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", {}) or {}
    forced_bos_token_id = lang_code_to_id.get(target_lang_code)
    if forced_bos_token_id is None:
        LOGGER.warning("Could not determine forced_bos_token_id for %s; generation may decode in the wrong language.", target_lang_code)

    dtype = get_dtype(model_cfg.get("torch_dtype"))
    base_model = MBartForConditionalGeneration.from_pretrained(
        model_cfg["name"],
        cache_dir=project_cfg.get("cache_dir"),
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    datasets = load_translation_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        cache_dir=project_cfg.get("cache_dir"),
    )
    if args.split not in datasets:
        raise ValueError(f"Split '{args.split}' not available in dataset. Available splits: {list(datasets.keys())}")
    dataset = datasets[args.split]

    records = build_records(dataset, data_cfg, args.num_samples, args.split)
    gen_kwargs = build_generation_kwargs(training_cfg, data_cfg)
    if forced_bos_token_id is not None and "forced_bos_token_id" not in gen_kwargs:
        # MBART needs the BOS token of the target language to guarantee correct generation.
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
    generate_outputs(model, tokenizer, records, device, args.batch_size, data_cfg, gen_kwargs)
    corpus_bleu = annotate_bleu_scores(records)
    write_csv(records, args.csv_path)

    LOGGER.info("Wrote %d rows to %s", len(records), args.csv_path)
    LOGGER.info("Corpus BLEU: %.2f", corpus_bleu)
    print(f"Corpus BLEU: {corpus_bleu:.2f}")


if __name__ == "__main__":
    main()
