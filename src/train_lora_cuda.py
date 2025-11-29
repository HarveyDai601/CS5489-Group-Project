"""CUDA-only LoRA fine-tuning entry point for WMT14 MT."""
from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data_utils import build_preprocess_function, load_translation_dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA-only LoRA fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_mt.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def get_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float16", "fp16"}:
        return torch.float16
    return torch.float32


def _require_grad_hook(_module, _inputs, output):
    if isinstance(output, torch.Tensor):
        output.requires_grad_(True)


def _ensure_embeddings_require_grad(target) -> None:
    embeddings = getattr(target, "get_input_embeddings", lambda: None)()
    if embeddings is not None and hasattr(embeddings, "weight"):
        embeddings.weight.requires_grad_(True)


def prepare_training_arguments(project_cfg: Dict[str, Any], training_cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(signature.parameters.keys())
    rename = {"evaluation_strategy": ("evaluation_strategy", "eval_strategy")}
    args_dict: Dict[str, Any] = {}

    for key, value in training_cfg.items():
        dest = key
        if key in rename:
            for candidate in rename[key]:
                if candidate in valid_params:
                    dest = candidate
                    break
            else:
                LOGGER.warning("Dropping unsupported training arg: %s", key)
                continue
        if dest in valid_params:
            args_dict[dest] = value
        else:
            LOGGER.warning("Dropping unsupported training arg: %s", dest)

    args_dict["output_dir"] = str(output_dir)
    if "logging_dir" in valid_params:
        args_dict["logging_dir"] = project_cfg.get("logging_dir", "runs")
    return args_dict


def build_split_overrides(data_cfg: Dict[str, Any], eval_base: str = "validation") -> Tuple[Dict[str, str], str]:
    def split_query(base: str, max_key: str) -> str:
        max_samples = data_cfg.get(max_key)
        if max_samples:
            return f"{base}[:{int(max_samples)}]"
        return base

    train_query = split_query("train", "max_train_samples")
    eval_query = split_query(eval_base, "max_eval_samples")
    overrides = {"train": train_query, eval_base: eval_query}
    return overrides, eval_base


def enable_gradient_checkpointing(model):
    base_model = getattr(getattr(model, "base_model", None), "model", None) or getattr(model, "model", None) or model
    base_model.gradient_checkpointing_enable()

    activated = False
    for target in (model, base_model):
        if hasattr(target, "enable_input_require_grads"):
            target.enable_input_require_grads()
            activated = True
    if not activated:
        embeddings = getattr(base_model, "get_input_embeddings", lambda: None)()
        if embeddings is not None:
            embeddings.register_forward_hook(_require_grad_hook)

    _ensure_embeddings_require_grad(base_model)

    if hasattr(model, "config"):
        model.config.use_cache = False

def find_latest_checkpoint(checkpoint_root):
    """
    Scan checkpoint_root for directories named like 'checkpoint-<num>' and return
    the path to the latest numeric checkpoint. Returns None if none found.
    """
    root = Path(checkpoint_root)
    if not root.exists():
        return None

    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not candidates:
        return None

    def _key(p: Path) -> int:
        parts = p.name.split("-")
        try:
            return int(parts[-1])
        except Exception:
            return -1

    latest = sorted(candidates, key=_key)[-1]
    return str(latest)

def main():
    configure_logging()
    if not torch.cuda.is_available():
        raise EnvironmentError("This script requires a CUDA-enabled GPU.")

    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir

    project_cfg = dict(config["project"])
    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    lora_cfg = dict(config["lora"])
    training_cfg = dict(config["training"])

    set_seed(project_cfg.get("seed", 42))

    overrides, eval_split = build_split_overrides(data_cfg, "validation")
    try:
        raw_datasets = load_translation_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config"),
            cache_dir=project_cfg.get("cache_dir"),
            split_overrides=overrides,
        )
    except ValueError:
        overrides, eval_split = build_split_overrides(data_cfg, "test")
        raw_datasets = load_translation_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config"),
            cache_dir=project_cfg.get("cache_dir"),
            split_overrides=overrides,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], cache_dir=project_cfg.get("cache_dir"))
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.src_lang = model_cfg["tokenizer_src_lang_code"]
    tokenizer.tgt_lang = model_cfg["tokenizer_tgt_lang_code"]

    dtype = get_dtype(model_cfg.get("torch_dtype", "float32"))
    model_kwargs: Dict[str, Any] = {"cache_dir": project_cfg.get("cache_dir"), "torch_dtype": dtype}

    if model_cfg.get("use_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs = {"cache_dir": project_cfg.get("cache_dir"), "quantization_config": quant_config, "device_map": "auto"}

    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["name"], **model_kwargs)

    if model_cfg.get("use_4bit"):
        model = prepare_model_for_kbit_training(model)

    if hasattr(model.config, "forced_bos_token_id") and hasattr(tokenizer, "lang_code_to_id"):
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[model_cfg["tokenizer_tgt_lang_code"]]

    task_type = TaskType.SEQ_2_SEQ_LM if lora_cfg["task_type"].lower() == "seq2seq_lm" else TaskType.SEQ_CLS
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if training_cfg.get("gradient_checkpointing"):
        enable_gradient_checkpointing(model)

    preprocess_fn = build_preprocess_function(tokenizer, data_cfg)
    remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    try:
        import evaluate

        bleu_metric = evaluate.load("sacrebleu")
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Falling back to eval_loss only because sacrebleu failed: %s", exc)
        bleu_metric = None
        desired_metric = training_cfg.get("metric_for_best_model")
        if desired_metric and desired_metric != "eval_loss":
            LOGGER.warning(
                "Overriding metric_for_best_model=%s to eval_loss because BLEU metrics are unavailable.",
                desired_metric,
            )
            training_cfg["metric_for_best_model"] = "eval_loss"

    def compute_metrics(eval_preds):
        if bleu_metric is None:
            return {}
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(labels, tuple):
            labels = labels[0]

        preds = np.asarray(preds)
        # Seq2SeqTrainer may hand us raw logits when generation is disabled; reduce to ids defensively.
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        if not np.issubdtype(preds.dtype, np.integer):
            preds = preds.astype(np.int64)
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        preds = np.where((preds >= 0) & (preds < vocab_size), preds, tokenizer.pad_token_id)

        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int64)
        labels = np.where((labels >= 0) & (labels < vocab_size), labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        return {"bleu": bleu["score"], "gen_len": np.mean(prediction_lens)}

    output_dir = Path(project_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(**prepare_training_arguments(project_cfg, training_cfg, output_dir))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets[eval_split],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    last_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        LOGGER.info("Resuming training from checkpoint: %s", last_checkpoint)
        train_result=trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(max_length=training_cfg["generation_max_length"], num_beams=training_cfg["generation_num_beams"])
    eval_metrics["eval_samples"] = len(processed_datasets[eval_split])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Training complete: %s", eval_metrics)


if __name__ == "__main__":
    main()
