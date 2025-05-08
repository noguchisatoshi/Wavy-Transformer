#!/usr/bin/env python3
# pretraining/pretrain.py
import argparse
import os
import csv
import math
import random
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from pretraining.pretraining_dataset import BertPretrainingDataset
from pretraining.model import CustomBertForPreTraining
from pretraining.config_loader import load_config

from utils.metrics import measure_over_smoothing, save_over_smoothing_image

def get_unique_filename(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    unique_path = file_path
    while os.path.exists(unique_path):
        unique_path = f"{base}_{counter}{ext}"
        counter += 1
    return unique_path

def parse_args():
    parser = argparse.ArgumentParser(description="Stable BERT Pretraining with Distributed Training")
    parser.add_argument("--config", type=str, default="configs/pretraining_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed training")
    parser.add_argument("--init_from_bert_base", action="store_true", help="Initialize model with pretrained bert-base weights")
    return parser.parse_args()

def print_config(config_dict):
    print("===== CONFIGURATION =====")
    print(json.dumps(config_dict, indent=4, ensure_ascii=False))
    print("=========================")

def evaluate_model(model, dataloader, device):
    model.eval()
    total_mlm_loss = 0.0
    total_masked_tokens = 0
    correct_predictions = 0
    
    vocab_size = model.module.bert.config.vocab_size if hasattr(model, "module") else model.bert.config.vocab_size

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            masked_lm_labels = batch["masked_lm_labels"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)

            with torch.amp.autocast("cuda"):
                loss, prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    masked_lm_labels=masked_lm_labels,
                    next_sentence_label=next_sentence_label
                )
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            mlm_loss = loss_fct(
                prediction_scores.view(-1, vocab_size),
                masked_lm_labels.view(-1)
            )
            total_mlm_loss += mlm_loss.item()

            mask = masked_lm_labels != -100
            num_masked_tokens = mask.sum().item()
            total_masked_tokens += num_masked_tokens

            if num_masked_tokens > 0:
                predictions = prediction_scores.argmax(dim=-1)
                correct_predictions += (predictions[mask] == masked_lm_labels[mask]).sum().item()

    avg_mlm_loss = total_mlm_loss / total_masked_tokens if total_masked_tokens > 0 else float("inf")
    perplexity = math.exp(avg_mlm_loss) if avg_mlm_loss < 300 else float("inf")
    accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0.0

    model.train()
    return perplexity, accuracy

def main():
    args = parse_args()
    config_dict = load_config(args.config)
    model_params = config_dict["model"]
    training_params = config_dict["training"]
    data_params = config_dict["data"]

    seed = config_dict.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    persistent_cache_path = data_params.get("persistent_cache_path", None)
    file_path = data_params["file_path"]

    max_seq_length = training_params["max_seq_length"]
    per_device_batch_size = training_params["per_device_batch_size"]
    gradient_accumulation_steps = training_params["gradient_accumulation_steps"]
    num_epochs = training_params["num_epochs"]
    learning_rate = training_params["learning_rate"]
    max_steps = training_params["max_steps"]
    eval_interval_steps = training_params.get("eval_interval_steps", 5000)

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    if rank == 0:
        print_config(config_dict)

    full_dataset = BertPretrainingDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        short_seq_prob=data_params["short_seq_prob"],
        masked_lm_prob=data_params["masked_lm_prob"],
        max_predictions_per_seq=data_params["max_predictions_per_seq"],
        dupe_factor=data_params["dupe_factor"],
        nsp_neg_prob=data_params["nsp_neg_prob"],
        persistent_cache_path=persistent_cache_path,
        max_documents=data_params["max_documents"]
    )

    total_samples = len(full_dataset)
    train_size = int(total_samples * 0.95)
    eval_size = total_samples - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size],
                                               generator=torch.Generator().manual_seed(seed))

    if config_dict.get("small_experiment", False):
        if rank == 0:
            print("Small experiment mode: Limiting train dataset to 100 samples.", flush=True)
        train_dataset = Subset(train_dataset, list(range(100)))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=per_device_batch_size, sampler=train_sampler)
    eval_dataloader = DataLoader(eval_dataset, batch_size=per_device_batch_size, shuffle=False)

    if rank == 0:
        print(f"Total full dataset samples: {total_samples}")
        print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        global_batch_size = per_device_batch_size * world_size
        estimated_batches_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
        estimated_global_steps_per_epoch = math.ceil(estimated_batches_per_epoch / gradient_accumulation_steps)
        print(f"Global batch size (per_device_batch_size x world_size): {global_batch_size}")
        print(f"Estimated batches per epoch: {estimated_batches_per_epoch}")
        print(f"Estimated global optimizer steps per epoch: {estimated_global_steps_per_epoch}")

    bert_config = BertConfig(
        vocab_size=model_params["vocab_size"],
        hidden_size=model_params["hidden_size"],
        num_hidden_layers=model_params["num_hidden_layers"],
        num_attention_heads=model_params["num_attention_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=model_params["max_position_embeddings"],
        hidden_dropout_prob=model_params["dropout_prob"],
        attention_probs_dropout_prob=model_params["dropout_prob"],
        layer_norm_eps=1e-12,
        num_labels=2
    )


    model = CustomBertForPreTraining(
        bert_config,
        residual_type=model_params["residual_type"],
        tau=model_params["tau"],
    ).to(device)

    if args.init_from_bert_base:
        from transformers import BertForPreTraining
        bert_pretrained = BertForPreTraining.from_pretrained("bert-base-uncased")
        model.bert.load_state_dict(bert_pretrained.bert.state_dict(), strict=False)
        if rank == 0:
            print("Pretrained bert-base weights have been loaded into the model.")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_warmup_steps = int(0.1 * max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=max_steps)
    scaler = torch.amp.GradScaler("cuda")

    initial_perplexity, initial_accuracy = evaluate_model(model, eval_dataloader, device)
    if rank == 0:
        print(f"[Initial Evaluation] Perplexity = {initial_perplexity:.2f}, MLM Accuracy = {initial_accuracy*100:.2f}%", flush=True)

    model.train()
    global_step = 0
    loss_history = []
    eval_history = []

    save_dir = training_params["save_dir"]
    log_save_dir = training_params["log_save_dir"]
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            masked_lm_labels = batch["masked_lm_labels"].to(device)
            next_sentence_label = batch["next_sentence_label"].to(device)

            with torch.amp.autocast("cuda"):
                loss, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    masked_lm_labels=masked_lm_labels,
                    next_sentence_label=next_sentence_label
                )
                mask = masked_lm_labels != -100
                num_masked_tokens = mask.sum().item()
                if num_masked_tokens == 0:
                    num_masked_tokens = 1
                loss = loss / num_masked_tokens
                loss = loss / gradient_accumulation_steps

            if torch.isnan(loss).item():
                if rank == 0:
                    print(f"NaN detected at Epoch: {epoch+1}, Step: {step+1}, Global Step: {global_step}. Stopping training.", flush=True)
                dist.barrier()
                dist.destroy_process_group()
                return

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if global_step > 0:
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_loss = loss.item() * gradient_accumulation_steps
                loss_history.append((epoch + 1, global_step, current_loss))

                if global_step % 20 == 0 and rank == 0:
                    print(f"Epoch: {epoch+1}, Global Step: {global_step}, Loss: {current_loss:.4f}", flush=True)

                if global_step % eval_interval_steps == 0:
                    perplexity, mlm_accuracy = evaluate_model(model, eval_dataloader, device)
                    if rank == 0:
                        print(f"[Evaluation] Global Step: {global_step}, Perplexity = {perplexity:.2f}, MLM Accuracy = {mlm_accuracy*100:.2f}%", flush=True)
                        eval_history.append((epoch+1, global_step, perplexity, mlm_accuracy))
                    
                    batch_eval = next(iter(eval_dataloader))
                    input_ids_eval = batch_eval["input_ids"].to(device)
                    attention_mask_eval = batch_eval["attention_mask"].to(device)
                    token_type_ids_eval = batch_eval["token_type_ids"].to(device)
                    masked_lm_labels_eval = batch_eval["masked_lm_labels"].to(device)
                    next_sentence_label_eval = batch_eval["next_sentence_label"].to(device)
                    
                    with torch.no_grad():
                        loss_eval, prediction_scores_eval, seq_relationship_score_eval, all_hidden_states, _ = model(
                            input_ids=input_ids_eval,
                            attention_mask=attention_mask_eval,
                            token_type_ids=token_type_ids_eval,
                            masked_lm_labels=masked_lm_labels_eval,
                            next_sentence_label=next_sentence_label_eval,
                            output_hidden_states=True,
                            output_attentions=False
                        )
                    
                    layerwise_means, layerwise_variances = measure_over_smoothing(all_hidden_states)
                    if rank == 0:
                        print(f"[Over-smoothing] Global Step: {global_step}", flush=True)
                        print(f"Layer-wise Mean Cosine Similarity: {layerwise_means}", flush=True)
                        print(f"Layer-wise Variance: {layerwise_variances}", flush=True)
                
                if global_step % 50000 == 0 and rank == 0:
                    bert_save_path = os.path.join(save_dir, f"{model_params['residual_type']}_pretrained_checkpoint_{global_step}.pt")
                    full_save_path = os.path.join(save_dir, f"{model_params['residual_type']}_full_model_checkpoint_{global_step}.pt")
                    torch.save(model.module.bert.state_dict(), bert_save_path)
                    torch.save(model.module.state_dict(), full_save_path)
                    print(f"Saved BERT model at step {global_step} to {bert_save_path}", flush=True)
                    print(f"Saved full model at step {global_step} to {full_save_path}", flush=True)

                if global_step >= max_steps:
                    if rank == 0:
                        print("Reached max training steps.", flush=True)
                        save_path = os.path.join(save_dir, f"{model_params['residual_type']}_base_model_checkpoint.pt")
                        torch.save(model.module.bert.state_dict(), save_path)
                        
                        os.makedirs(log_save_dir, exist_ok=True)
                        loss_csv_path = os.path.join(log_save_dir, f"{model_params['residual_type']}_loss_history.csv")
                        loss_csv_path = get_unique_filename(loss_csv_path)
                        with open(loss_csv_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["epoch", "global_step", "loss"])
                            writer.writerows(loss_history)
                        
                        eval_csv_path = os.path.join(log_save_dir, f"{model_params['residual_type']}_eval_history.csv")
                        eval_csv_path = get_unique_filename(eval_csv_path)
                        with open(eval_csv_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["epoch", "global_step", "perplexity", "mlm_accuracy"])
                            writer.writerows(eval_history)
                    dist.barrier()
                    dist.destroy_process_group()
                    return

    if rank == 0:
        print("Pretraining finished!", flush=True)
        os.makedirs(log_save_dir, exist_ok=True)
        loss_csv_path = os.path.join(log_save_dir, f"{model_params['residual_type']}_loss_history.csv")
        loss_csv_path = get_unique_filename(loss_csv_path)
        with open(loss_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "global_step", "loss"])
            writer.writerows(loss_history)
            
        eval_csv_path = os.path.join(log_save_dir, f"{model_params['residual_type']}_eval_history.csv")
        eval_csv_path = get_unique_filename(eval_csv_path)
        with open(eval_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "global_step", "perplexity", "mlm_accuracy"])
            writer.writerows(eval_history)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()