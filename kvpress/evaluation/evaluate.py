# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
import time
from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from experiments.pythia_wikitext_perplexity import compute_perplexity
from kvpress import (
    ComposedPress,
    DecodingPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    ThinKPress,
)

logger = logging.getLogger(__name__)

PERPLEXITY_DATASETS = {
    "wikitext_ppl": {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-103-v1",
        "split": "test",
        "text_column": "text",
        "max_samples": 64,
        "use_streaming": False,
        "max_eval_tokens": 4096,
    },
    "pg19_ppl": {
        "dataset_name": "pg19",
        "dataset_config": None,
        "split": "test",
        "text_column": "text",
        "max_samples": 1,
        "use_streaming": True,
        "max_eval_tokens": 8192,
    },
}


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None

    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    compress_questions: bool = False
    needle_depth: Optional[int] = None
    ppl_max_length: int = 1024
    ppl_stride: int = 512
    press_sequence: list[str] = field(default_factory=list)
    prefill_only: bool = False

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Validate compression ratios
        assert (
            0.0 <= self.compression_ratio <= 1.0
        ), f"compression_ratio must be between 0.0 and 1.0, got {self.compression_ratio}"

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

        if self.press_sequence:
            for name in self.press_sequence:
                assert name in PRESS_REGISTRY, f"Press '{name}' not found in PRESS_REGISTRY"

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path
        press
            The press instance to check for ThinKPress components

        Returns
        -------
        Path
            The path to the results directory
        """
        # Build directory name components
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.compress_questions:
            components.append("compressed_questions")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # Make sure the directory does not exist, if it does, add a number to the end
        # This is to avoid overwriting results
        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        with open(str(config_filename), "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self.perplexity_texts: list[str] = []
        self.is_perplexity_dataset = self.config.dataset in PERPLEXITY_DATASETS
        self._cached_tokenizer = None
        self._cached_model = None
        self._cached_input_ids = None
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _build_results_dir_for_press(self, output_dir: Path, press_name: str, compression_ratio: float) -> Path:
        components = [
            self.config.dataset,
            str(self.config.data_dir) if self.config.data_dir else "",
            self.config.model.replace("/", "--"),
            press_name,
            f"{compression_ratio:.2f}",
        ]
        if self.config.fraction < 1.0:
            components.append(f"fraction{self.config.fraction:.3f}")
        if self.config.max_context_length is not None:
            components.append(f"max_context{self.config.max_context_length}")
        if self.config.compress_questions:
            components.append("compressed_questions")
        dir_name = "__".join(filter(None, components))
        results_dir = output_dir / dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]
        if press is not None:
            press = copy.deepcopy(press)

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            press.head_compression_ratio = compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {compression_ratio}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, DecodingPress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        if self.is_perplexity_dataset:
            self._load_perplexity_dataset()
            return

        logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
        df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        if isinstance(self.press, FinchPress):
            if not self.config.compress_questions:
                logger.error("FinchPress requires 'compress_questions' to be set to True.")
                raise ValueError("FinchPress requires compress_questions to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(
                self.pipeline.model, self.pipeline.tokenizer
            )  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.compress_questions:
            logger.info("Compressing questions into context.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _load_perplexity_dataset(self):
        dataset_name = self.config.dataset
        cfg = PERPLEXITY_DATASETS[dataset_name]
        dataset_kwargs = {"split": cfg["split"], "trust_remote_code": True}
        if cfg.get("use_streaming"):
            dataset_kwargs["streaming"] = True
        hf_args = [cfg["dataset_name"]]
        if cfg.get("dataset_config"):
            hf_args.append(cfg["dataset_config"])

        logger.info(f"Loading perplexity dataset from Hugging Face: {cfg['dataset_name']} ({cfg.get('dataset_config')})")
        dataset = load_dataset(*hf_args, **dataset_kwargs)
        texts: list[str] = []
        max_samples = cfg.get("max_samples")
        for idx, row in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            text = row.get(cfg["text_column"], "")
            if text and not text.isspace():
                texts.append(text)

        if not texts:
            raise ValueError(f"Unable to load any text samples for {dataset_name}.")

        self.perplexity_texts = texts
        self.df = pd.DataFrame({"text": texts})
        logger.info(f"Collected {len(texts)} samples for perplexity evaluation.")

    def _get_perplexity_model_inputs(self) -> tuple[AutoModelForCausalLM, torch.Tensor, torch.device, torch.dtype]:
        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        if self._cached_input_ids is None:
            cfg = PERPLEXITY_DATASETS[self.config.dataset]
            tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            concatenated = "\n\n".join(self.perplexity_texts)
            input_ids = tokenizer(concatenated, return_tensors="pt").input_ids
            max_eval_tokens = cfg.get("max_eval_tokens")
            if max_eval_tokens:
                input_ids = input_ids[:, :max_eval_tokens]
            logger.info(f"Tokenized corpus contains {input_ids.shape[1]} tokens.")
            self._cached_tokenizer = tokenizer
            self._cached_input_ids = input_ids

        if self._cached_model is None or self._cached_device != device:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)
            model.eval()
            self._cached_model = model
            self._cached_device = device
            self._cached_dtype = dtype

        return self._cached_model, self._cached_input_ids, device, dtype

    def _setup_model_pipeline(self):
        if self.is_perplexity_dataset:
            logger.info("Perplexity dataset detected, skipping pipeline setup.")
            self.pipeline = None
            return
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}
        if isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """
        if self.is_perplexity_dataset:
            logger.info("Perplexity dataset detected, computing perplexity directly.")
            self._run_perplexity_inference()
            return

        self.df["predicted_answer"] = None  # type: ignore[index]

        if isinstance(self.press, DecodingPress):
            logger.info("DecodingPress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                questions = df_group["question"].to_list()
                # Use max_new_tokens from config, or fallback to dataset's default for the task
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]

                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                self.df.loc[df_group.index, "predicted_answer"] = output["answers"]  # type: ignore[union-attr]
                # Store the actual compression ratio used (if the press has one)
                self.df.loc[df_group.index, "compression_ratio"] = (
                    self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                )  # type: ignore[union-attr, attr-defined]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        logger.info("Inference completed.")

    def _run_perplexity_inference(self):
        if not self.perplexity_texts:
            raise ValueError("Perplexity texts not loaded.")

        model, encoded_dataset, device, _ = self._get_perplexity_model_inputs()

        ppl, total_time, prefill_time = compute_perplexity(
            model,
            encoded_dataset,
            device=device,
            max_length=self.config.ppl_max_length,
            stride=self.config.ppl_stride,
            press=self.press,
            track_time=True,
        )

        runtime = prefill_time if self.config.prefill_only else total_time
        compression_ratio = (
            getattr(self.press, "compression_ratio", 0.0) if self.press is not None else 0.0
        )

        self.df = pd.DataFrame(
            [
                {
                    "dataset": self.config.dataset,
                    "press": self.config.press_name,
                    "perplexity": ppl,
                    "runtime_sec": runtime,
                    "total_runtime_sec": total_time,
                    "prefill_runtime_sec": prefill_time,
                    "compression_ratio": compression_ratio,
                }
            ]
        )
        logger.info(
            f"Perplexity evaluation complete: ppl={ppl:.4f}, "
            f"runtime={runtime:.3f}s (prefill={prefill_time:.3f}s, compression_ratio={compression_ratio:.2f})"
        )
        return ppl, total_time, prefill_time

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)  # type: ignore[call-arg]

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        if self.config.press_sequence:
            self._run_press_sequence(output_dir)
            return

        results_dir = self.config.get_results_dir(output_dir)
        predictions_filename = results_dir / "predictions.csv"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        logger.info("Evaluation run completed successfully.")

    def _run_press_sequence(self, output_dir: Path):
        if not self.is_perplexity_dataset:
            raise ValueError("Press sequence evaluation is only supported for perplexity datasets.")

        original_press = self.config.press_name
        original_ratio = self.config.compression_ratio

        self._load_and_prepare_dataset()

        for idx, press_name in enumerate(self.config.press_sequence, start=1):
            logger.info(f"Sequential press {idx}/{len(self.config.press_sequence)}: {press_name}")
            self.config.press_name = press_name
            if press_name == "no_press":
                self.config.compression_ratio = 0.0
            else:
                self.config.compression_ratio = original_ratio
            self._setup_press()
            self._run_perplexity_inference()

            results_dir = self._build_results_dir_for_press(
                output_dir, press_name, self.config.compression_ratio
            )
            predictions_filename = results_dir / "predictions.csv"
            metrics_filename = results_dir / "metrics.json"
            config_filename = results_dir / "config.yaml"

            self._save_results(predictions_filename)
            self._calculate_and_save_metrics(metrics_filename)
            self.config.save_config(config_filename)

        self.config.press_name = original_press
        self.config.compression_ratio = original_ratio
        logger.info("Sequential press evaluation completed.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)
        seq = final_args.get("press_sequence")
        if isinstance(seq, str):
            final_args["press_sequence"] = [s.strip() for s in seq.split(",") if s.strip()]
        elif seq is None:
            final_args["press_sequence"] = []
        prefill_flag = final_args.get("prefill_only")
        if isinstance(prefill_flag, str):
            final_args["prefill_only"] = prefill_flag.lower() in {"1", "true", "yes", "y"}

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
