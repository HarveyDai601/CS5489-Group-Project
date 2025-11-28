"""Utility functions for loading and preprocessing translation datasets."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_translation_dataset(
    dataset_name: str,
    dataset_config: str | None = None,
    cache_dir: str | None = None,
    split_overrides: Mapping[str, str] | None = None,
) -> DatasetDict:
    """Loads the requested translation dataset from the Hugging Face Hub."""

    if split_overrides:
        dataset = DatasetDict()
        for split_name, split_query in split_overrides.items():
            dataset[split_name] = load_dataset(
                dataset_name,
                dataset_config,
                split=split_query,
                cache_dir=cache_dir,
            )
        return dataset

    return load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)


def build_preprocess_function(
    tokenizer: PreTrainedTokenizerBase,
    data_cfg: Dict[str, Any],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Creates a preprocessing function that tokenizes translation pairs."""

    prompt_template = data_cfg["prompt_template"]
    source_language = data_cfg["source_language"]
    target_language = data_cfg["target_language"]
    text_column = data_cfg["text_column"]
    max_source_length = data_cfg["max_source_length"]
    max_target_length = data_cfg["max_target_length"]

    def preprocess_function(batch: Dict[str, Any]) -> Dict[str, Any]:
        translations = batch[text_column]
        inputs = []
        targets = []

        for example in translations:
            source_text = example[source_language]
            target_text = example[target_language]
            prompt = prompt_template.format(
                source_lang=source_language,
                target_lang=target_language,
                source_text=source_text,
            )
            inputs.append(prompt)
            targets.append(target_text)

        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function
