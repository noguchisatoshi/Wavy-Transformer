# ImageNet object classification

It is based on DeiT [official repository](https://github.com/facebookresearch/deit) and the FeatScale [official repository](https://github.com/VITA-Group/ViT-Anti-Oversmoothing).

## Getting Started

### Dependency

First of all, clone our repository locally:

```
git clone https://github.com/noguchisatoshi/Wavy-Transformer.git
```

Then, install the following Python libraries which are required to run our code:

```
pytorch 1.7.0
cudatoolkit 11.0
torchvision 0.8.0
timm 0.4.12
```

### Data Preparation

Download and extract ImageNet train and val images from the [official website](http://image-net.org/).
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

To automatically collate the dataset directory, you may find these [shell scripts](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) useful.

## Usage

### Training

Training Wavy Transformer from scratch usually requires multiple GPUs. Please use the following command to train our model with distributed data parallel:

```
python -m torch.distributed.launch --nproc_per_node=<num_nodes> --master_port <port> --use_env \
main.py --auto_reload --model <model_name> --batch-size <batch_size> \
--data-path <data_path> --data-set IMNET --input-size 224 \
--output_dir <log_dir>
```
where `<model_name>` specifies the name of model to build, such as 'tiny_12_wave', 'featscale_tiny_12_wave'.

To reproduce our results, please follow the command lines below:

<details>

<summary>
12-layer DeiT-Tiny + Wavy Transformer
</summary>

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29700 --use_env \
main.py --auto_reload --model tiny_12_wave --batch-size 256 --clip-grad 1.0 \
--data-path </data_path> --data-set IMNET --input-size 224 \
--output_dir ./logs/imnet1k_tiny_12_wave
```

</details>

<details>

<summary>
12-layer DeiT-Tiny + Wavy Transformer + FeatScale
</summary>

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29700 --use_env \
main.py --auto_reload --model featscale_tiny_12_wave --batch-size 256 --clip-grad 1.0 \
--data-path </data_path> --data-set IMNET --input-size 224 \
--output_dir ./logs/imnet1k_featscale_tiny_12_wave
```
</details>


### Pre-trained Models

Our pre-trained model parameters are included in model_parameters directory. To evaluate our pre-trained models, please specify flags `--eval` and `--resume` to the path to the checkpoints. For example, to reproduce our results of `DeiT-Tiny + Wavy Transformer + FeatScale`, one can run the following command:
```
python main.py --model featscale_tiny_12_wave
--eval --resume </ckpt_dir>/tiny_12_wave+featscale.pth  --data-path </data_path> --data-set IMNET
```

### Oversmoothing analysis

To analyze oversmoothing behavior of our pre-trained models, please use oversmoothing.py. For example, to reproduce our results of `DeiT-Tiny + Wavy Transformer + FeatScale`, one can run the following command:
```
  python oversmoothing.py --data-set IMNET --data-path </data_path> \
    --model featscale_tiny_12_wave --resume </ckpt_dir>/tiny_12_wave+featscale.pth  \
    --batch-size 64 --num-workers 8 --pin-mem \
    --use-all --output-dir .log/oversmooth_results/featscale_tiny_12_wave
```
## Model parameters
- [DeiT-Tiny + Wave Transformer](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/vit/tiny_12_wave.pth?download=true)
- [DeiT-Tiny + Wavy Transformer + FeatScale](https://huggingface.co/ngtsts/Wavy-Transformer/resolve/main/vit/tiny_12_wave%2Bfeatscale.pth?download=true)