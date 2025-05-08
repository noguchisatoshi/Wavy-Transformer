# Wavy Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the **official implementation** of **Wavy Transformer**&nbsp;<!-- TODO: add arXiv/DOI link when available -->.  

---

## Introduction

Transformers have achieved remarkable success in both natural language processing (NLP) and computer vision (CV). However, deep transformer models can suffer from **over-smoothing**, where token representations converge to similar values as they pass through successive blocks.  

**Wavy Transformer** mitigates this issue by introducing:

* a novel attention layer based on **second-order wavy dynamics**,  
* a feed-forward network and normalization layer that preserve the physical state–velocity relationship implied by the chain rule.

Across diverse NLP and CV benchmarks, Wavy Transformer consistently improves performance **with minimal extra parameters and no additional hyper-parameter tuning**.

<p align="center">
  <img src="figures/wavy_block.png" width="650" alt="Wavy block illustration">
</p>

---

## Contents

This repository comprises two main components:

* **NLP Tasks**: Everything related to pretraining the BERT-base model, fine-tuning on downstream benchmarks (GLUE and SQ2AD), and analyzing oversmoothing behavior.
* **CV Tasks**: Scripts and examples for ImageNet object classification using Vision Transformers, including training, evaluation, and analyzing oversmoothing behavior.

## Citation

A formal citation will be provided here as soon as our paper is publicly available (arXiv / conference proceedings, currently in preparation). In the meantime, if you find *Wavy Transformer* useful for your research or applications, please consider pointing to this repository.

<!-- BibTeX entry will appear here after submission -->



