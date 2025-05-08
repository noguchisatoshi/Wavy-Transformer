#!/usr/bin/env python3
"""
Refactored main.py for training and evaluation of CustomBertForSequenceClassification.
This version follows GLUE-baselines evaluation metrics and, for MNLI, evaluates both
matched and mismatched sets, recording both scores. Overall performance across tasks is computed 
and saved in the experiments/results folder.
"""

import argparse
import csv
import json
import logging
import random
import copy
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import tqdm
import scipy.stats
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

from models.bert import CustomBertForSequenceClassification
from utils.metrics import compute_accuracy, measure_over_smoothing, save_over_smoothing_image
from utils.get_logging import get_logger
from data.dataset import get_dataset
from transformers import BertConfig


def setup_logger(log_file: str) -> logging.Logger:
    """Sets up the logger with a file handler."""
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Training and Evaluation for Custom BERT")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                        help="Run mode: train or evaluate")
    parser.add_argument("--use_bert_base", action="store_true", 
                        help="If set, load model from bert-base-uncased pretrained weights.")
    parser.add_argument(
        "--random_seeds", "-rs",
        type=int, nargs="+", default=None,
        help="List of random seeds to run (e.g. `-rs 0 1 2 3`). "
             "If not provided, uses the seed in the config file."
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a JSON file and sets up experiment name."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for field in ["data_type", "residual_type", "tau", "random_seed"]:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    dt = config["data_type"]
    rt = config["residual_type"]
    tau = config["tau"]
    exp_name = f"{rt}_experiment_{dt}_tau={tau}"
    config["experiment_name"] = exp_name
    return config


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model(args: argparse.Namespace, config: Dict[str, Any],
                     device: torch.device, logger: logging.Logger,
                     skip_load: bool = False) -> torch.nn.Module:
    rt = config.get("residual_type", "diffuse")
    dt = config["data_type"]
    base_dt = "mnli" if dt in ("mnli_matched", "mnli_mismatched") else dt

    if base_dt == "stsb":
        config["num_labels"] = 1
    elif base_dt == "mnli":
        config["num_labels"] = 3
    elif base_dt in {"cola", "sst2", "mrpc", "qqp", "qnli", "rte", "wnli"}:
        config["num_labels"] = 2
    else:
        config["num_labels"] = 2

    model_cfg = BertConfig(
        vocab_size=config.get("vocab_size", 30522),
        hidden_size=config.get("hidden_size", 768),
        intermediate_size=config.get("intermediate_size", 3072),
        num_hidden_layers=config.get("num_hidden_layers", 12),
        num_attention_heads=config.get("num_attention_heads", 12),
        max_position_embeddings=config.get("max_position_embeddings", 512),
        hidden_dropout_prob=config.get("dropout_prob", 0.1),
        num_labels=config["num_labels"]
    )

    if config.get("reference_model", False):
        model = CustomBertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            residual_type=rt,
            num_labels=config["num_labels"]
        )
    else:
        model = CustomBertForSequenceClassification(
            model_cfg,
            residual_type=rt,
            tau=config["tau"]
        )

    model.to(device)

    if args.mode == "evaluate" and not skip_load:
        exp_name = config["experiment_name"]
        if dt in ("mnli_matched", "mnli_mismatched"):
            exp_name = exp_name.replace(dt, "mnli")

        if "result_dir" in config:
            ckpt = (
                Path("./experiments")
                / config["result_dir"]
                / exp_name
                / "checkpoints"
                / "best_model.pt"
            )
        else:
            ckpt = (
                Path("./experiments")
                / f"{rt}_experiment_{base_dt}_tau={config['tau']}"
                / "checkpoints"
                / "best_model.pt"
            )

        checkpoint = torch.load(ckpt, map_location=device)
        load_result = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded checkpoint from {ckpt}")
        logger.info(f"Missing keys: {load_result.missing_keys}")
        logger.info(f"Unexpected keys: {load_result.unexpected_keys}")

    return model


def compute_glue_metrics(preds: np.ndarray, labels: np.ndarray, data_type: str) -> Tuple[float, Dict[str, float]]:
    """Computes GLUE-baselines metrics for the given task."""
    if data_type == "cola":
        mcc = matthews_corrcoef(labels, preds)
        return mcc, {"matthews_corrcoef": mcc}
    elif data_type in {"mrpc", "qqp"}:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        avg = (acc + f1) / 2
        return avg, {"accuracy": acc, "f1": f1, "avg": avg}
    elif data_type == "stsb":
        p = scipy.stats.pearsonr(labels, preds)[0]
        s = scipy.stats.spearmanr(labels, preds)[0]
        avg = (p + s) / 2
        return avg, {"pearson": p, "spearman": s, "avg": avg}
    else:
        acc = accuracy_score(labels, preds)
        return acc, {"accuracy": acc}


def compute_combined_metrics(model: torch.nn.Module,
                             loader: torch.utils.data.DataLoader,
                             device: torch.device,
                             data_type: str,
                             num_layers: int) -> Tuple[Tuple[float, Dict[str, float]], Tuple[List[float], List[float]]]:
    """
    Performs a forward pass per batch to compute both standard GLUE metrics
    and oversmoothing metrics (cosine similarity and variance per layer).
    """
    model.eval()
    all_preds, all_labels = [], []
    sims_all = [[] for _ in range(num_layers)]
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            n_batches += 1
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if data_type == "stsb":
                labels = labels.float()
            outputs = model(input_ids, attn, output_hidden_states=True)
            logits = outputs[1]
            hiddens = outputs[2]
            if data_type == "stsb":
                preds = logits.squeeze(dim=-1).cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            sims, _ = measure_over_smoothing(hiddens[1:])  # Exclude embedding layer.
            for i in range(num_layers):
                sims_all[i].append(sims[i])
    std_metric, std_details = compute_glue_metrics(np.array(all_preds), np.array(all_labels), data_type)
    avg_sims = [np.mean(layer_sims) for layer_sims in sims_all]
    var_sims_overall = [np.var(layer_sims) for layer_sims in sims_all]
    return (std_metric, std_details), (avg_sims, var_sims_overall)


def save_oversmoothing_results(avg_sims: List[float],
                               avg_vars: List[float],
                               exp_dir: Path,
                               logger: logging.Logger) -> None:
    """Saves oversmoothing results as CSV and generates an image."""
    csv_path = exp_dir / "average_sims.csv"
    with csv_path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "cosine similarity", "variance"])
        for i, (s, v) in enumerate(zip(avg_sims, avg_vars)):
            writer.writerow([i, s, v])
    save_over_smoothing_image(avg_sims, avg_vars, exp_dir)
    logger.info("Oversmoothing results saved.")


def evaluate_model(model: torch.nn.Module,
                   loader: torch.utils.data.DataLoader,
                   config: Dict[str, Any],
                   device: torch.device,
                   logger: logging.Logger) -> None:
    """
    Evaluates the model by computing both standard GLUE and oversmoothing metrics.
    Saves oversmoothing results.
    """
    dt = config["data_type"]
    num_layers = config["num_hidden_layers"]
    (metric, details), (avg_sims, avg_vars) = compute_combined_metrics(model, loader, device, dt, num_layers)
    logger.info(f"Evaluation ({dt}) - Metric: {metric:.4f}, Details: {details}")
    logger.info(f"Oversmoothing - Avg cosine similarity: {avg_sims}")
    logger.info(f"Oversmoothing - Avg variance: {avg_vars}")
    if "result_dir" in config:
        exp_dir = Path("experiments") / config["result_dir"] / config["experiment_name"] / "results"
    else:
        exp_dir = Path("experiments") / config["experiment_name"] / "results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_oversmoothing_results(avg_sims, avg_vars, exp_dir, logger)


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          config: Dict[str, Any],
          device: torch.device,
          logger: logging.Logger) -> None:
    """Training loop with scheduler and weight decay. Saves progress to CSV and plots progress."""
    logger.info("Starting training...")
    if "result_dir" in config:
        exp_dir = Path("experiments") / config["result_dir"] / config["experiment_name"]
    else:
        exp_dir = Path("experiments") / config["experiment_name"]
    csv_path = exp_dir / "training_progress.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "avg_loss", "val_metric"])

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.01}
    ]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"])
    num_epochs = config.get("num_epochs", 3)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, max(0, (total_steps - step) / (total_steps - warmup_steps)))
    )

    best_score = float("inf") if config["data_type"] == "stsb" else float("-inf")
    best_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if config["data_type"] == "stsb":
                labels = labels.float()
            outputs = model(input_ids, attn, labels=labels, output_hidden_states=True)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        (val_metric, val_details), _ = compute_combined_metrics(model, val_loader, device, config["data_type"], config["num_hidden_layers"])
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Metric={val_metric:.4f}")
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch+1, f"{avg_loss:.4f}", f"{val_metric:.4f}"])
        better = val_metric < best_score if config["data_type"] == "stsb" else val_metric > best_score
        if better:
            best_score = val_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    if best_state is not None:
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best_model.pt"
        torch.save(best_state, ckpt_path)
        logger.info(f"Best model saved to {ckpt_path} at Epoch {best_epoch+1}")
    else:
        logger.warning("No improvement during training.")


def evaluate_tasks(args: argparse.Namespace,
                   config: Dict[str, Any],
                   device: torch.device,
                   logger: logging.Logger) -> Tuple[Dict[str, float], Dict[str, Any]]:
    tasks = config.get("glue_tasks",
                       ["cola", "sst2", "mrpc", "qqp", "mnli_matched", "mnli_mismatched",
                        "qnli", "rte", "wnli", "stsb"])
    detailed = {}
    overall_macro, overall_micro, total_examples = 0.0, 0.0, 0

    for task in tasks:
        local_cfg = copy.deepcopy(config)
        local_cfg["data_type"] = task
        local_cfg["batch_size"] = 1

        base_task = "mnli" if task in ("mnli_matched", "mnli_mismatched") else task
        local_cfg["experiment_name"] = (
            f"{local_cfg['residual_type']}_experiment_{base_task}_tau={local_cfg['tau']}"
        )

        _, loader = get_dataset(local_cfg)
        model = initialize_model(args, local_cfg, device, logger)
        (metric, details), _ = compute_combined_metrics(
            model, loader, device, task, local_cfg["num_hidden_layers"]
        )

        n_ex = sum(batch["label"].size(0) for batch in loader)
        detailed[task] = {"metric": metric, "details": details, "n_examples": n_ex}
        overall_macro += metric
        overall_micro += metric * n_ex
        total_examples += n_ex

        logger.info(f"Task {task}: {metric:.4f} over {n_ex} examples")

    overall_macro /= len(tasks)
    overall_micro = overall_micro / total_examples if total_examples else 0
    overall = {"macro_accuracy": overall_macro, "micro_accuracy": overall_micro}
    if "result_dir" in config:
        exp_dir = Path("experiments") / config["result_dir"] / config["experiment_name"] / "results"
    else:
        exp_dir = Path("experiments") / config["experiment_name"] / "results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    overall_path = exp_dir / "overall_results.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump({"overall_metrics": overall, "detailed": detailed}, f, indent=4)
    logger.info(f"Overall results saved to {overall_path}")
    return overall, detailed

def evaluate_across_seeds(
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    seed_list: List[int]
) -> Dict[str, float]:

    rt         = config["residual_type"]
    tau        = config["tau"]
    result_dir = config.get("result_dir", "")

    tasks = config.get("glue_tasks", ["cola","sst2","mrpc","qqp","mnli","qnli","rte","wnli","stsb"])
    avg_metrics: Dict[str, float] = {}

    for task in tasks:
        base_task = "mnli" if task in ("mnli_matched", "mnli_mismatched") else task
        base_name = f"{rt}_experiment_{base_task}_tau={tau}"

        scores: List[float] = []
        for seed in seed_list:
            set_seed(seed)
            exp_name = f"{base_name}_seed{seed}"

            cfg = copy.deepcopy(config)
            cfg["data_type"] = task
            cfg["experiment_name"] = exp_name
            _, loader = get_dataset(config=cfg)

            model = initialize_model(args, cfg, device, logger, skip_load=True)

            ckpt_path = (
                Path("experiments")
                / result_dir
                / exp_name
                / "checkpoints"
                / "best_model.pt"
            )
            logger.info(f"[{task}][seed={seed}] Loading {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model_dict = model.state_dict()
            filtered = {
                k: v for k, v in state.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            model.load_state_dict(filtered, strict=False)
            model.to(device)

            (metric, _), _ = compute_combined_metrics(
                model, loader, device, task, cfg["num_hidden_layers"]
            )
            logger.info(f"[{task}][seed={seed}] metric = {metric:.4f}")
            scores.append(metric)

        avg = sum(scores) / len(scores)
        avg_metrics[task] = float(avg)
        logger.info(f"[{task}] average over seeds {seed_list} = {avg:.4f}")

    return avg_metrics


def run_single_task(args: argparse.Namespace,
                    config: Dict[str, Any],
                    device: torch.device,
                    logger: logging.Logger) -> None:
    """
    Runs a single task for training or evaluation.
    In evaluation mode, enforces batch_size=1 and uses dynamic checkpoint paths.
    """
    train_loader, val_loader = get_dataset(config=config)
    model = initialize_model(args, config, device, logger)
    task = config["data_type"]
    mode = args.mode
    logger.info(f"===== Start {mode.upper()} for task: {task} =====")

    if args.mode == "train":
        train(model, train_loader, val_loader, config, device, logger)
    else:
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_loader.dataset, batch_size=1, shuffle=False, collate_fn=val_loader.collate_fn)
        evaluate_model(model, val_loader, config, device, logger)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    if args.use_bert_base:
        config["reference_model"]   = True
        config["experiment_name"] = f"base_experiment_{config['data_type']}"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_list = args.random_seeds or [config["random_seed"]]

    if args.mode == "evaluate":
        logger = setup_logger("evaluate.log")
        logger.info(f"Evaluating tasks {config.get('glue_tasks')} over seeds {seed_list}")

        avg_metrics = evaluate_across_seeds(args, config, device, logger, seed_list)
        out_dir = (
            Path("experiments")
            / config.get("result_dir", config["experiment_name"])
            / "results"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        save_path = out_dir / "all_tasks_avg_metrics.json"
        with save_path.open("w", encoding="utf-8") as f:
            json.dump({"average_metrics": avg_metrics}, f, indent=2)

        logger.info(f"Saved all-tasks average metrics to {save_path}")
        return

    base_name = config["experiment_name"]
    for seed in seed_list:
        config["random_seed"]   = seed
        config["experiment_name"] = f"{base_name}_seed{seed}"
        set_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        logger = setup_logger(f"output_seed{seed}.log")
        logger.info(f"Running seed={seed}, mode={args.mode}")

        if config.get("run_all_glue", False):
            default_lr     = config["learning_rate"]
            default_epochs = config.get("num_epochs", 3)
            small_tasks    = {"cola", "stsb", "mrpc", "rte", "wnli"}
            for task in config.get("glue_tasks"):
                cfg = copy.deepcopy(config)
                cfg["data_type"] = task
                if task in small_tasks:
                    cfg["learning_rate"], cfg["num_epochs"] = 5e-5, 3
                else:
                    cfg["learning_rate"], cfg["num_epochs"] = default_lr, default_epochs
                cfg["experiment_name"] = (
                    f"{cfg['residual_type']}_experiment_{task}_tau={cfg['tau']}" + f"_seed{seed}"
                )
                run_single_task(args, cfg, device, logger)
        else:
            run_single_task(args, config, device, logger)


if __name__ == "__main__":
    main()