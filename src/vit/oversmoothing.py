#!/usr/bin/env python3
# oversmooth.py

import os
import argparse
import csv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import build_dataset
from timm.models import create_model
import models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run oversmoothing diagnostics (cosine similarity & variances) using forward-hooks on block outputs")
    parser.add_argument('--data-path',   required=True,
                        help='root path to dataset (for ImageNet, path to “imagenet_root”)')
    parser.add_argument('--data-set',    default='IMNET',
                        choices=['CIFAR','IMNET','IMNET100','INAT','INAT19'],
                        help='which dataset to use')
    parser.add_argument('--model',       default='deit_base_patch16_224',
                        help='timm model name or custom @register_model name')
    parser.add_argument('--resume',      default=None, type=str,
                        help='path to model checkpoint (.pth) to load')
    parser.add_argument('--input-size',  type=int, default=224,
                        help='input image size for transforms')
    parser.add_argument('--batch-size',  type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--pin-mem',     action='store_true', default=True,
                        help='pin_memory for DataLoader')
    parser.add_argument('--max-batches', type=int, default=50,
                        help='how many batches to run for diagnostics (ignored if --use-all)')
    parser.add_argument('--use-all',     action='store_true', default=False,
                        help='process all validation batches instead of stopping at --max-batches')
    parser.add_argument('--device',      default='cuda',
                        help='device to run on')
    parser.add_argument('--output-dir',  default='./',
                        help='where to save the plots and data')
    return parser.parse_args()


def run_oversmooth(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Building dataset ({args.data_set}) from {args.data_path}...")
    val_dataset, nb_classes = build_dataset(is_train=False, args=args)
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    total_batches = len(loader) if args.use_all else args.max_batches

    print(f"Instantiating model {args.model} (num_classes={nb_classes})...")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=nb_classes,
    )

    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded.")

    model.to(device).eval()

    # Hook: capture block outputs (exclude CLS token)
    features = []
    def hook_block(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out  # [B, N, D]
        features.append(x[:, 1:, :].detach().cpu())    # drop CLS

    for blk in model.blocks:
        blk.register_forward_hook(hook_block)

    print("Running through validation batches...")
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(loader, total=total_batches, desc="Batches")):
            imgs = imgs.to(device)
            _ = model(imgs)
            if not args.use_all and i + 1 >= args.max_batches:
                break

    # Metric functions
    def avg_cosine_sim(x):
        xn = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sims = torch.einsum('bid,bjd->bij', xn, xn)
        N = x.shape[1]
        mask = ~torch.eye(N, dtype=torch.bool)
        sims_flat = sims[:, mask].view(x.shape[0], -1)
        return sims_flat.mean(dim=1), sims_flat.var(dim=1)

    print("Computing metrics per layer...")
    L = len(features) // total_batches
    feats = [
        torch.cat([features[b * L + i] for b in range(total_batches)], dim=0)
        for i in range(L)
    ]

    Cs_mean = []
    Cs_var  = []
    for f in feats:
        mean_vals, var_vals = avg_cosine_sim(f)
        Cs_mean.append(mean_vals.mean().item())
        Cs_var.append(var_vals.mean().item())

    # Save CSV: layer, cos similarity mean, cos similarity var
    csv_path = os.path.join(args.output_dir, 'oversmooth_data.csv')
    print(f"Saving raw data to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'cos_mean', 'cos_var'])
        for idx, (m, v) in enumerate(zip(Cs_mean, Cs_var)):
            writer.writerow([idx, m, v])

    # Plotting
    layers = list(range(L))

    print("Plotting cosine similarity mean...")
    plt.figure()
    plt.plot(layers, Cs_mean, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Avg pairwise cosine similarity')
    plt.title('Oversmoothing: Similarity rise (mean)')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'oversmooth_cosine_mean.png'))
    plt.close()

    print("Plotting cosine similarity variance...")
    plt.figure()
    plt.plot(layers, Cs_var, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Variance of pairwise cosine similarity')
    plt.title('Oversmoothing: Similarity distribution variance')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'oversmooth_cosine_var.png'))
    plt.close()

    print(f"Oversmoothing data and plots saved in {args.output_dir}")

if __name__ == '__main__':
    args = parse_args()
    run_oversmooth(args)