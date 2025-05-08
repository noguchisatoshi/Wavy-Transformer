#!/usr/bin/env python3
# analyze_oversmoothing.py

import os
import sys
import csv
import math
import torch
import argparse
import yaml
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.metrics import (
    compute_cosine_similarity
    save_over_smoothing_image,
    save_energy_transition_image,
    compute_dirichlet_energy,
    compute_avg_energy
)

from models.bert import CustomBertForSequenceClassification

def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    encodings = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt")
    return encodings

def compute_cosine_similarity_per_sample(hs):
    norm = torch.norm(hs, dim=-1, keepdim=True) + 1e-8
    hs_norm = hs / norm  # (B, S, D)
    cos_sim_matrix = torch.bmm(hs_norm, hs_norm.transpose(1, 2))  # (B, S, S)
    B, S, _ = cos_sim_matrix.shape
    diag_mask = 1 - torch.eye(S, device=hs.device).unsqueeze(0).expand(B, -1, -1)
    cos_sim_matrix = cos_sim_matrix * diag_mask
    avg_cos = cos_sim_matrix.sum(dim=(1,2)) / (S * (S - 1))
    return avg_cos

def analyze_over_smoothing_cosine(model, tokenizer, dataset, batch_size, device, output_dir, residual_type, args):
    model.eval()
    num_layers = model.bert.config.num_hidden_layers  # without embeddings
    layer_means_list = [[] for _ in range(num_layers)]
    
    worst_cos = -float("inf")
    worst_sample_hidden_states = None  
    worst_sample_input_ids = None     
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            all_hidden_states = outputs[2] 
            for i, hs in enumerate(all_hidden_states):
                if i > 0:
                    mean_sim, _ = compute_cosine_similarity(hs)
                    layer_means_list[i-1].append(mean_sim)
            final_hs = all_hidden_states[-1] 
            batch_cos = compute_cosine_similarity_per_sample(final_hs)
            max_val, max_idx = torch.max(batch_cos, dim=0)
            if max_val.item() > worst_cos:
                worst_cos = max_val.item()
                worst_sample_hidden_states = [hs[max_idx].unsqueeze(0).cpu() for hs in all_hidden_states]
                worst_sample_input_ids = input_ids[max_idx].cpu()
    
    final_means = [sum(layer_means_list[i]) / len(layer_means_list[i]) for i in range(num_layers)]
    layer_means_variance = []
    for i in range(num_layers):
        means_tensor = torch.tensor(layer_means_list[i])
        if means_tensor.numel() > 1:
            layer_var = means_tensor.var(unbiased=True).item()
        else:
            layer_var = 0.0
        layer_means_variance.append(layer_var)

    csv_path = os.path.join(output_dir, f"{residual_type}_over_smoothing_with_variance.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["layer", "cos similarity", "variance"])
        for i, (mean_val, var_val) in enumerate(zip(final_means, layer_means_variance)):
            writer.writerow([i, mean_val, var_val])
    print(f"CSV saved to {csv_path}")
    
    save_over_smoothing_image(final_means, layer_means_variance, output_dir, residual_type)
    
    if worst_sample_hidden_states is not None:
        worst_history = []
        for hs in worst_sample_hidden_states:
            sample_cos = compute_cosine_similarity_per_sample(hs)
            worst_history.append(sample_cos.item())
        worst_text = tokenizer.decode(worst_sample_input_ids, skip_special_tokens=True)
        worst_csv_path = os.path.join(output_dir, f"{residual_type}_worst_case_cos_sim_history.csv")
        with open(worst_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["layer", "cos similarity"])
            for i, cos_val in enumerate(worst_history):
                writer.writerow([i, cos_val])
        print(f"Worst-case cosine similarity history CSV saved to {worst_csv_path}")
        
        worst_plot_path = os.path.join(output_dir, f"{residual_type}_worst_case_cos_sim_history.png")
        layers = range(1, len(worst_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(layers, worst_history, marker='o', color='blue', label='Worst-case Cosine Similarity')
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.title('Worst-case Cosine Similarity History', fontsize=14)
        plt.grid(True)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(worst_plot_path)
        plt.close()
        print(f"Worst-case cosine similarity plot saved to {worst_plot_path}")
        
        if len(worst_sample_hidden_states) > 1:
            worst_hidden_states_energy = torch.stack(worst_sample_hidden_states[1:], dim=1)
            worst_energy = compute_dirichlet_energy(worst_hidden_states_energy)
            worst_energy = worst_energy.cpu().numpy()
            worst_energy_csv_path = os.path.join(output_dir, f"{residual_type}_worst_case_energy_history.csv")
            with open(worst_energy_csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["layer", "energy"])
                for i, energy_val in enumerate(worst_energy):
                    writer.writerow([i+1, energy_val])
            print(f"Worst-case energy history CSV saved to {worst_energy_csv_path}")
            
            worst_energy_plot_path = os.path.join(output_dir, f"{residual_type}_worst_case_energy_history.png")
            layers = range(1, len(worst_energy) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(layers, worst_energy, marker='o', color='red', label='Worst-case Energy')
            plt.xlabel('Layer', fontsize=12)
            plt.ylabel('Energy', fontsize=12)
            plt.title('Worst-case Energy History', fontsize=14)
            plt.grid(True)
            plt.legend()
            plt.savefig(worst_energy_plot_path)
            plt.close()
            print(f"Worst-case energy plot saved to {worst_energy_plot_path}")
        else:
            print("No hidden states beyond embedding for energy computation.")
    else:
        print("No worst-case sample found.")

def analyze_over_smoothing_energy(model, tokenizer, dataset, batch_size, device, output_dir, residual_type, tau, energy_metric, args):
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    total_energy_sum = None
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches (energy)", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if energy_metric == "avg":
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
                all_hidden_states = outputs[2]
                all_attentions = outputs[3]
                intra_attn_weights = torch.stack([attn.mean(dim=1) for attn in all_attentions], dim=1)
            else:
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False)
                all_hidden_states = outputs[2]
                intra_attn_weights = None
            hidden_states_layers = torch.stack(all_hidden_states, dim=1)
            if energy_metric == "avg":
                batch_energy = compute_avg_energy(hidden_states_layers, intra_attn_weights, tau)
            else:
                batch_energy = compute_dirichlet_energy(hidden_states_layers)
            batch_energy = batch_energy.cpu()
            B_current = hidden_states_layers.size(0)
            if total_energy_sum is None:
                total_energy_sum = batch_energy * B_current
            else:
                total_energy_sum += batch_energy * B_current
            total_samples += B_current
    final_avg_energy = total_energy_sum / total_samples
    csv_path = os.path.join(output_dir, f"{residual_type}_energy_transition.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["layer", "energy"])
        for i, energy_val in enumerate(final_avg_energy.numpy()):
            writer.writerow([i, energy_val])
    print(f"Energy CSV saved to {csv_path}")
    save_energy_transition_image(final_avg_energy, output_dir, residual_type)

def parse_max_documents(value):
    if value.lower() == "full":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("max_documents must be an integer or 'full'.")

def main():
    parser = argparse.ArgumentParser(description="Analyze over-smoothing in BERT models using text data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--residual_type", type=str, required=True,
                        choices=["diffuse", "wave", "mix", "wave_simp", "mix_simp"],
                        help="Residual type to use for the custom model")
    parser.add_argument("--max_documents", type=parse_max_documents, default=10000,
                        help="Maximum number of documents to process. Use 'full' to process all documents.")
    parser.add_argument("--metric", type=str, choices=["cosine", "energy"], default=None,
                        help="If specified, only run the chosen metric; otherwise run both.")
    parser.add_argument("--energy_metric", type=str, choices=["dirichlet", "avg"], default="dirichlet",
                        help="Energy metric to use: 'dirichlet' uses compute_dirichlet_energy; 'avg' uses compute_avg_energy")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r") as f:
        config_yaml = yaml.safe_load(f)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # --- Model configuration from YAML ---
    model_params = config_yaml["model"]
    config_dict = {
        "vocab_size": model_params.get("vocab_size", 30522),
        "hidden_size": model_params.get("hidden_size", 768),
        "intermediate_size": model_params.get("intermediate_size", 3072),
        "num_hidden_layers": model_params.get("num_hidden_layers", 12),
        "num_attention_heads": model_params.get("num_attention_heads", 12),
        "max_position_embeddings": model_params.get("max_position_embeddings", 512),
        "hidden_dropout_prob": model_params.get("hidden_dropout_prob", 0.1),
        "attention_probs_dropout_prob": model_params.get("attention_probs_dropout_prob", 0.1),
        "num_labels": model_params.get("num_labels", 2)
    }
    model_config = BertConfig(**config_dict)
    
    # --- Initialize Model ---
    tau = model_params.get("tau", 0.5)
    pretrain_model_path = model_params.get("pretrain_model_path", None)
    checkpoint_path = f"./finetune_models/finetune_model_{args.residual_type}_24_v2.pt"
        
    model = CustomBertForSequenceClassification(
        model_config,
        residual_type=args.residual_type,
        tau=tau,
    )

    checkpoint_state_dict = torch.load(checkpoint_path, map_location=device)
    model_state_dict = model.bert.state_dict()
    filtered_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            if value.size() == model_state_dict[key].size():
                filtered_state_dict[key] = value
            else:
                logger.info(f"Skipping parameter '{key}' due to shape mismatch: checkpoint {value.size()} vs model {model_state_dict[key].size()}")
        else:
            logger.info(f"Parameter '{key}' not found in current model.")

    model.bert.load_state_dict(filtered_state_dict, strict=False)
    logger.info(f"Loaded custom BERT model from {checkpoint_path}")
    
    model.to(device)
    
    # --- Data ---
    text_file = config_yaml["data"].get("text_file")
    max_documents = config_yaml["data"].get("max_documents", 10000)
    
    if not os.path.exists(text_file):
        logger.error(f"Text file not found: {text_file}")
        return

    with open(text_file, "r", encoding="utf-8") as f:
        data = f.read()
    documents = data.split("\n\n")
    if max_documents == "full":
        documents = documents
    elif max_documents is not None:
        documents = documents[:max_documents]
    
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": documents})
    
    # --- Output directory ---
    output_dir = config_yaml["output"].get("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Run analysis ---
    if args.metric is None:
        analyze_over_smoothing_cosine(
            model, tokenizer, dataset, batch_size=1,
            device=device, output_dir=output_dir, residual_type=args.residual_type, args=args
        )
        analyze_over_smoothing_energy(
            model, tokenizer, dataset, batch_size=1,
            device=device, output_dir=output_dir, residual_type=args.residual_type,
            tau=tau, energy_metric=args.energy_metric, args=args
        )
    elif args.metric == "cosine":
        analyze_over_smoothing_cosine(
            model, tokenizer, dataset, batch_size=1,
            device=device, output_dir=output_dir, residual_type=args.residual_type, args=args
        )
    elif args.metric == "energy":
        analyze_over_smoothing_energy(
            model, tokenizer, dataset, batch_size=1,
            device=device, output_dir=output_dir, residual_type=args.residual_type,
            tau=tau, energy_metric=args.energy_metric, args=args
        )

if __name__ == "__main__":
    main()