"""Prepare WMT14 translation pairs for LlamaFactory SFT training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LlamaFactory SFT dataset from WMT14 translations")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_mt.yaml",
        help="Path to the existing YAML config that holds dataset metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="llamafactory_sft/data",
        help="Directory where the JSONL splits will be written.",
    )
    parser.add_argument(
        "--dataset_stem",
        type=str,
        default="wmt14_en_de",
        help="Base file name used for the exported JSONL files (e.g., <stem>_train.jsonl).",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Name of the training split to export.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        help="Name of the evaluation split to export (falls back to test if missing).",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def materialize_split(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    cache_dir: str | None,
) -> Dataset:
    try:
        return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    except ValueError as exc:
        raise ValueError(f"Unable to load split '{split}' from {dataset_name}:{dataset_config}") from exc


def build_prompt(prompt_template: str, source_lang: str, target_lang: str, source_text: str) -> str:
    return prompt_template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
    ).strip()


def convert_to_conversations(example: Dict[str, Any], data_cfg: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    translation = example[data_cfg["text_column"]]
    source_text = translation[data_cfg["source_language"]]
    target_text = translation[data_cfg["target_language"]]

    prompt = build_prompt(prompt_template, data_cfg["source_language"], data_cfg["target_language"], source_text)

    return {
        "id": example.get("id"),
        "source": f"{data_cfg['dataset_name']}:{data_cfg.get('dataset_config', 'default')}",
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_text.strip()},
        ],
    }


def export_split(
    dataset: Dataset,
    split_name: str,
    limit: int | None,
    data_cfg: Dict[str, Any],
    prompt_template: str,
    target_path: Path,
):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target_path.open("w", encoding="utf-8") as sink:
        for idx, example in enumerate(dataset):
            if limit is not None and idx >= limit:
                break
            record = convert_to_conversations(example, data_cfg, prompt_template)
            record["id"] = record.get("id") or f"{split_name}-{idx}"
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    project_cfg = config["project"]
    data_cfg = config["data"].copy()
    data_cfg["dataset_name"] = data_cfg.get("dataset_name", "wmt/wmt14")

    prompt_template = data_cfg["prompt_template"]
    cache_dir = project_cfg.get("cache_dir")

    train_dataset = materialize_split(data_cfg["dataset_name"], data_cfg.get("dataset_config"), args.train_split, cache_dir)
    eval_split = args.eval_split
    try:
        eval_dataset = materialize_split(data_cfg["dataset_name"], data_cfg.get("dataset_config"), eval_split, cache_dir)
    except ValueError:
        eval_split = "test"
        eval_dataset = materialize_split(data_cfg["dataset_name"], data_cfg.get("dataset_config"), eval_split, cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_limit = data_cfg.get("max_train_samples")
    eval_limit = data_cfg.get("max_eval_samples")

    train_path = output_dir / f"{args.dataset_stem}_{args.train_split}.jsonl"
    eval_path = output_dir / f"{args.dataset_stem}_{eval_split}.jsonl"

    train_count = export_split(train_dataset, args.train_split, train_limit, data_cfg, prompt_template, train_path)
    eval_count = export_split(eval_dataset, eval_split, eval_limit, data_cfg, prompt_template, eval_path)

    print(f"Wrote {train_count} training samples to {train_path}")
    print(f"Wrote {eval_count} evaluation samples to {eval_path}")


if __name__ == "__main__":
    main()
