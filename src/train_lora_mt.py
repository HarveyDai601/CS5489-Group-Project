"""LoRA fine-tuning script for WMT14 machine translation."""
from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict

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
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for WMT14 translation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_mt.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for the output directory set in the config.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def get_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16" or name == "fp16":
        return torch.float16
    return torch.float32


def detect_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def prepare_training_arguments(
    project_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Builds training arguments dict compatible with the installed transformers version."""

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(signature.parameters.keys())
    rename_candidates = {
        "evaluation_strategy": ("evaluation_strategy", "eval_strategy"),
    }

    args_dict: Dict[str, Any] = {}
    for key, value in training_cfg.items():
        dest_key = key
        if key in rename_candidates:
            for candidate in rename_candidates[key]:
                if candidate in valid_params:
                    dest_key = candidate
                    break
            else:
                LOGGER.warning("Dropping training argument '%s' because it is unsupported.", key)
                continue
        if dest_key in valid_params:
            args_dict[dest_key] = value
        else:
            LOGGER.warning("Dropping unsupported training argument '%s'.", dest_key)

    args_dict["output_dir"] = str(output_dir)
    if "logging_dir" in valid_params:
        args_dict["logging_dir"] = project_cfg.get("logging_dir", "runs")
    return args_dict


def enable_gradient_checkpointing(model):
    base_model = getattr(getattr(model, "base_model", None), "model", model)
    base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    else:
        input_embeddings = base_model.get_input_embeddings()
        if input_embeddings is not None:
            def make_inputs_require_grad(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)
    if hasattr(model, "config"):
        model.config.use_cache = False


def main():
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir

    project_cfg = dict(config["project"])
    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    lora_cfg = dict(config["lora"])
    training_cfg = dict(config["training"])
    training_cfg.setdefault("dataloader_pin_memory", True)

    set_seed(project_cfg.get("seed", 42))

    device_type = detect_device_type()
    LOGGER.info("Detected compute backend: %s", device_type)

    if model_cfg.get("use_4bit") and device_type != "cuda":
        LOGGER.warning("4-bit quantization requires CUDA; disabling it for %s backend.", device_type)
        model_cfg["use_4bit"] = False

    # Ensure dtype/training precision align with available hardware
    dtype = get_dtype(model_cfg.get("torch_dtype", "float32"))
    if device_type != "cuda" and dtype in (torch.float16, torch.bfloat16):
        LOGGER.warning("Falling back to float32 weights on %s backend.", device_type)
        dtype = torch.float32

    if device_type != "cuda":
        if training_cfg.get("bf16"):
            LOGGER.info("Disabling bf16 on %s backend.", device_type)
            training_cfg["bf16"] = False
        if training_cfg.get("fp16"):
            LOGGER.info("Disabling fp16 on %s backend.", device_type)
            training_cfg["fp16"] = False
        if training_cfg.get("gradient_checkpointing"):
            LOGGER.info("Disabling gradient checkpointing on %s backend.", device_type)
            training_cfg["gradient_checkpointing"] = False
            model_cfg["gradient_checkpointing"] = False
        if training_cfg.get("dataloader_num_workers", 0) > 0:
            LOGGER.info("Setting dataloader_num_workers=0 for %s backend to avoid shared-memory issues.", device_type)
            training_cfg["dataloader_num_workers"] = 0
        if training_cfg.get("dataloader_pin_memory", True):
            LOGGER.info("Disabling dataloader_pin_memory on %s backend.", device_type)
            training_cfg["dataloader_pin_memory"] = False

    cache_dir = project_cfg.get("cache_dir")

    def build_split_query(split_base: str, max_samples_key: str) -> str:
        max_samples = data_cfg.get(max_samples_key)
        if max_samples:
            return f"{split_base}[:{int(max_samples)}]"
        return split_base

    train_split_query = build_split_query("train", "max_train_samples")
    eval_split_base = "validation"
    eval_split_query = build_split_query(eval_split_base, "max_eval_samples")

    split_overrides = {"train": train_split_query, eval_split_base: eval_split_query}
    try:
        raw_datasets = load_translation_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config"),
            cache_dir=cache_dir,
            split_overrides=split_overrides,
        )
        eval_split_name = eval_split_base
    except ValueError:
        eval_split_base = "test"
        eval_split_query = build_split_query(eval_split_base, "max_eval_samples")
        split_overrides = {"train": train_split_query, eval_split_base: eval_split_query}
        raw_datasets = load_translation_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config"),
            cache_dir=cache_dir,
            split_overrides=split_overrides,
        )
        eval_split_name = eval_split_base

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.src_lang = model_cfg["tokenizer_src_lang_code"]
    tokenizer.tgt_lang = model_cfg["tokenizer_tgt_lang_code"]

    model_kwargs: Dict[str, Any] = {"cache_dir": cache_dir}

    if model_cfg.get("use_4bit"):
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

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

    preprocess_function = build_preprocess_function(tokenizer, data_cfg)
    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = None
    try:
        import evaluate

        metric = evaluate.load("sacrebleu")
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Falling back to metric-less training because evaluate.load failed: %s", exc)

    def compute_metrics(eval_preds):
        if metric is None:
            return {}
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        bleu = metric.compute(predictions=decoded_preds, references=decoded_labels)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result = {"bleu": bleu["score"], "gen_len": np.mean(prediction_lens)}
        return result

    output_dir = Path(project_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args_dict = prepare_training_arguments(project_cfg, training_cfg, output_dir)
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets[eval_split_name],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(max_length=training_cfg["generation_max_length"], num_beams=training_cfg["generation_num_beams"])
    eval_metrics["eval_samples"] = len(processed_datasets[eval_split_name])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Training complete. Metrics: %s", eval_metrics)


if __name__ == "__main__":
    main()
