''''

# GLUE task and oversmoothing for BERT

## Getting Started

### Dependency

First, clone our repository locally:

```bash
git clone https://github.com/noguchisatoshi/Wavy-Transformer.git
```

Then, install the required Python libraries:

```text
transformers
torch
datasets
```

## Data Preparation

1. **Download and process pretraining data**

   We provide a script `pretraining/download_dataset.py` to automatically download and preprocess the BooksCorpus and Wikipedia datasets. It saves the combined documents to `datasets/pretraining/data.txt`.

   ```bash
   python pretraining/download_dataset.py --max_documents 100000
   ```

2. **Prepare GLUE/SQ2AD data**

    Both GLUE and SQ2AD datasets are loaded directly via the Hugging Face datasets library—no manual download required.

## Usage

### 1. Pre-training

Pre-train BERT-base from scratch using `pretrain.py`:

````bash
torchrun --master_port=29700 --nproc_per_node=2 pretrain.py --config pretraining/configs_pretrain/config_diffuse.yaml
````

### 2. Fine-tuning on GLUE

Fine-tune on GLUE tasks using `run_glue.py`, where `-rs` is used specify random seeds:

<details>
```bash
python run_glue.py --config glue_configs/config_diffuse_GLUE.json --mode train -rs 41 42 43
````
</details>

### 3. Fine-tuning with SQuAD

Fine-tune on SQuAD with `finetune_SQuAD/finetune.py`:

```bash
python finetune_SQuAD/finetune.py --config_file finetune_SQuAD/config.yaml
```

### 4. Oversmoothing Analysis

Analyze oversmoothing behavior using `analyze_over_smoothing/analyze_over_smoothing.py`:

```bash
python analyze_over_smoothing/analyze_over_smoothing.py --config analyze_over_smoothing/config.yaml
```
## Model parameters
### Pretraining models for each residual connection
- [Diffuse pretrain model](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/bert/diffuse_base_model_checkpoint.pt?download=true)
- [Wave pretrain model](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/bert/wave_base_model_checkpoint.pt?download=true)
- [Mix pretrain model](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/bert/mix_base_model_checkpoint.pt?download=true)

### Finetune model for each residual connection
- [Diffuse finetune model](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/bert/finetune_SQuAD/finetune_model_diffuse.pt?download=true)
- [Wave finetune model](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/bert/finetune_SQuAD/finetune_model_wave.pt?download=true)
- [Mix finetune model](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/bert/finetune_SQuAD/finetune_model_mix.pt?download=true)


## Scripts Explanation

* `pretrain.py`: scripts for masked language model pre-training.
* `run_glue.py`: scripts for fine-tuning on GLUE benchmark.
* `finetune.py`: scripts for fine-tuning on SQ2AD dataset.
* `analyze_over_smoothing.py`: compute pairwise representation similarities across layers.