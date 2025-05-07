# Wavy Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of [Wavy Transformer]().


This repository is built based on DeiT [official repository](https://github.com/facebookresearch/deit) and FeatScale [official repository](https://github.com/VITA-Group/ViT-Anti-Oversmoothing).

## Introduction

Vision Transformer (ViT) has recently demonstrated promise in computer vision problems. However, unlike Convolutional Neural Networks (CNN), it is known that the performance of ViT saturates quickly with depth increasing, due to the observed attention collapse or patch uniformity. Despite a couple of empirical solutions, a rigorous framework studying on this scalability issue remains elusive. In this paper, we first establish a  rigorous theory framework to analyze ViT features from the Fourier spectrum domain. We show that the self-attention mechanism inherently amounts to a low-pass filter, which indicates when ViT scales up its depth, excessive low-pass filtering will cause feature maps to only preserve their Direct-Current (DC) component. We then propose two straightforward yet effective techniques to mitigate the undesirable low-pass limitation. The first technique, termed AttnScale, decomposes a self-attention block into low-pass and high-pass components, then rescales and combines these two filters to produce an all-pass self-attention matrix. The second technique, termed FeatScale, re-weights feature maps on separate frequency bands to amplify the high-frequency signals. Both techniques are efficient and hyperparameter-free, while effectively overcoming relevant ViT training artifacts such as attention collapse and patch uniformity. By seamlessly plugging in our techniques to multiple ViT variants, we demonstrate that they consistently help ViTs benefit from deeper architectures, bringing up to 1.1% performance gains "for free" (e.g., with little parameter overhead).

<p align="center">
  <img src="figures/wavy_block.png" width="500">
</p>

## Getting Started

### Dependency

First of all, clone our repository locally:

```
git clone https://github.com/VITA-Group/ViT-Anti-Oversmoothing.git
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


## Citation

If you find this work or our code implementation helpful for your own resarch or work, please cite our paper.
```

```
