# PIO-FVLM

This repository is the official implementation of [PIO-FVLM]().

**[PIO-FVLM: Rethinking Training-Free Visual Token Reduction for VLM Acceleration from an Inference-Objective Perspective]()**
<br/>
[Haokui Zhang](https://scholar.google.com/citations?hl=en&user=m3gPwCoAAAAJ), 
[Congyang Ou](https://github.com/ocy1), 
[Dawei Yan](https://scholar.google.com/citations?user=U8KJSfcAAAAJ&hl=zh-CN&oi=ao), 
[Peng Wang](), 
[Qingsen Yan](https://scholar.google.com/citations?user=BSGy3foAAAAJ&hl=zh-CN&oi=ao), 
[Ying Li](), 
[Rong Xiao](), 
[Chunhua Shen](https://scholar.google.com/citations?hl=en&user=Ljk2BvIAAAAJ)
<br/>

## News
- [02/01/2026] Code will be released soon!


## Overall
<p align="center">
  <a href="https://raw.githubusercontent.com/ocy1/PIO-FVLM/main/Images/overall_new.pdf">
    <img src="https://raw.githubusercontent.com/ocy1/PIO-FVLM/main/Images/overall_new.png" alt="Structure" width="100%" />
  </a>
  <br/>
  <em>
   
  </em>
</p>

<blockquote>
  <p>
    </strong> We propose PIO-FVLM, a training-free method that selects vision tokens via gradient saliency and a feature-space NMS strategy, improving efficiency while preserving performance and maintaining compatibility with efficient attention operators.
  </p>
</blockquote>

## Example
<p align="center">
  <a href="https://raw.githubusercontent.com/ocy1/PIO-FVLM/main/Images/4%20methods%20new.pdf">
    <img src="https://raw.githubusercontent.com/ocy1/PIO-FVLM/main/Images/4%20methods.jpg"
         alt="Comparison of token selection strategies"
         width="92%" />
  </a>
  <br/>
  <em>
    Comparison of different token selection strategies.
  </em>
</p>

<blockquote>
  <p>
   </strong> Comparison of different token selection strategies.
    (a) Ours; (b) text-to-image attention based; (c) diversity-oriented selection;
    (d) CLS-token attention based.
  </p>
</blockquote>

## Comprehensive Benchmark Results

The following table presents a detailed comparison of our method (PIO-FVLM) against various state-of-the-art baselines across multiple benchmarks under different token retention budgets.

| Type (Tokens) | Methods | Venue | GQA | MMB | MMB-cn | MME | POPE | SQA | VQAv2 | TextVQA | Average |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Upper Bound (576)** | Vanilla | – | 61.9 | 64.7 | 58.1 | 1862 | 85.9 | 69.5 | 78.5 | 58.2 | 100% |
| | | | | | | | | | | | |
| **w/o VE (192, 33.3%)** | FastV | ECCV’24 | 52.7 | 61.2 | 57.0 | 1612 | 64.8 | 67.3 | 67.1 | 52.5 | 89.0% |
| | PDrop | CVPR’25 | 57.1 | 63.2 | 56.8 | 1766 | 82.3 | 68.8 | 75.1 | 56.1 | 96.2% |
| | SparseVLM | ICML’25 | 59.5 | 64.1 | 53.7 | 1787 | 85.3 | 68.7 | 75.6 | 57.8 | 97.2% |
| | DART | EMNLP’25 | 60.0 | 63.6 | 57.0 | 1856 | 82.8 | 69.8 | 76.7 | 57.4 | 98.3% |
| | **PIO-FVLM (Ours)** | **Ours** | **61.0** | **64.4** | **57.6** | **1789** | **86.5** | **69.0** | **77.7** | **57.2** | **98.8%** |
| | | | | | | | | | | | |
| **w/ VE (192, 33.3%)**| VisionZip | CVPR’25 | 59.3 | 63.0 | 57.3 | 1782 | 85.3 | 68.9 | 76.8 | 57.3 | 97.8% |
| | HoloV | NeurIPS’25 | 59.0 | 65.4 | 58.0 | 1820 | 85.6 | 69.8 | 76.7 | 57.4 | 98.7% |
| | SCOPE | NeurIPS’25 | 60.1 | 63.6 | 56.8 | 1804 | 86.4 | 68.8 | 77.2 | 57.7 | 98.3% |
| | **PIO-FVLM (Ours)** | **Ours** | **61.1** | **64.2** | **57.9** | **1808** | **86.4** | **68.2** | **77.9** | **57.4** | **98.9%** |
| | | | | | | | | | | | |
| **w/o VE (128, 22.2%)**| FastV | ECCV’24 | 49.6 | 56.1 | 56.4 | 1490 | 59.6 | 60.2 | 61.8 | 50.6 | 83.2% |
| | PDrop | CVPR’25 | 56.0 | 61.1 | 56.6 | 1644 | 82.3 | 68.3 | 72.9 | 55.1 | 94.0% |
| | SparseVLM | ICML’25 | 58.4 | 64.5 | 51.1 | 1746 | 85.0 | 68.6 | 73.8 | 56.7 | 95.6% |
| | DART | EMNLP’25 | 58.7 | 63.2 | 57.5 | 1840 | 80.1 | 69.1 | 75.9 | 56.4 | 97.0% |
| | **PIO-FVLM (Ours)** | **Ours** | **60.0** | **62.9** | **57.1** | **1807** | **86.7** | **68.5** | **76.5** | **57.2** | **98.1%** |
| | | | | | | | | | | | |
| **w/ VE (128, 22.2%)**| VisionZip | CVPR’25 | 57.6 | 62.0 | 56.7 | 1761.7 | 83.2 | 68.9 | 75.6 | 56.8 | 96.4% |
| | HoloV | NeurIPS’25 | 57.7 | 63.9 | 56.5 | 1802 | 84.0 | 69.8 | 75.5 | 56.8 | 97.2% |
| | SCOPE | NeurIPS’25 | 59.7 | 62.5 | 56.9 | 1776 | 86.1 | 68.4 | 76.5 | 57.2 | 97.5% |
| | **PIO-FVLM (Ours)** | **Ours** | **60.0** | **62.9** | **56.7** | **1799** | **86.4** | **69.2** | **77.1** | **57.0** | **98.1%** |
| | | | | | | | | | | | |
| **w/o VE (64, 11.1%)** | FastV | ECCV’24 | 46.1 | 48.0 | 52.7 | 1256 | 48.0 | 51.1 | 55.0 | 47.8 | 74.0% |
| | PDrop | CVPR’25 | 41.9 | 33.3 | 50.5 | 1092 | 55.9 | 68.6 | 69.2 | 45.9 | 74.4% |
| | SparseVLM | ICML’25 | 53.8 | 60.1 | 52.7 | 1589 | 77.5 | 69.8 | 68.2 | 53.4 | 90.6% |
| | DART | EMNLP’25 | 55.9 | 60.6 | 53.2 | 1765 | 73.9 | 69.8 | 72.4 | 54.4 | 92.8% |
| | **PIO-FVLM (Ours)** | **Ours** | **58.0** | **61.6** | **53.7** | **1681** | **84.3** | **68.5** | **74.8** | **54.9** | **94.7%** |
| | | | | | | | | | | | |
| **w/ VE (64, 11.1%)** | VisionZip | CVPR’25 | 55.1 | 60.1 | 55.4 | 1690 | 77.0 | 69.0 | 72.4 | 55.5 | 93.0% |
| | HoloV | NeurIPS’25 | 55.3 | 63.3 | 55.1 | 1715 | 80.3 | 69.5 | 72.8 | 55.4 | 94.4% |
| | SCOPE | NeurIPS’25 | 58.3 | 61.7 | 54.4 | 1698 | 83.9 | 68.6 | 75.3 | 56.6 | 95.4% |
| | **PIO-FVLM (Ours)** | **Ours** | **58.3** | **61.6** | **56.5** | **1744** | **86.4** | **68.6** | **75.9** | **56.2** | **96.6%** |

> **Note:** Base model used is **LLaVA-1.5-7B**. The baseline "Upper Bound" utilizes all 576 tokens.
## Performance Results

Here is the performance comparison of our method against the baseline LLaVA models. Our approach significantly reduces computation overhead (Prefill Time, Total Time, FLOPs, and KV Cache) while maintaining competitive performance on the POPE benchmark.

| Methods | Prefill Time↓ (s) | Total Time↓ (s) | Avg FLOPs↓ (T) | KV Cache↓ (MB) | POPE (Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LLaVA-1.5-7B** | 1401 (1.00×) | 2234 (1.00×) | 2.98 (1.00×) | 318 (1.00×) | 85.9 |
| +Ours(%11.1) | **1106 (1.27×)** | **1978 (1.13×)** | **0.45 (6.62×)** | **62 (5.13×)** | 84.3 |
| **LLaVA-NEXT-7B** | 4934 (1.00×) | 5921 (1.00×) | 16.67 (1.00×) | 1156 (1.00×) | 86.5 |
| +Ours(%11.1) | **1844 (2.67×)** | **2810 (2.11×)** | **2.68 (6.22×)** | **191 (6.05×)** | 84.5 |

> **Note:** > * `↓` indicates that lower is better. 
> * Values in parentheses represent the reduction ratio/speedup compared to the respective baseline.

## Set Up
## LLaVA

1. Clone this repository.

```bash
git clone https://github.com/ocy1/PIO-FVLM
cd PIO-FVLM
```

2.Environment Setup and Preparation

```conda create -n PIO_FVLM python=3.10 -y
conda activate PIO_FVLM
pip install -e .
pip install flash-attn --no-build-isolation
```
3.Download Multimodal Benchmark
Please follow the detailed instruction in https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

## Qwen2.5-VL
```
conda create -n PIO_FVLM_Qwen25VL python=3.10 -y
conda activate PIO_FVLM_Qwen25VL

pip install -U transformers==4.55.4
pip install flash-attn --no-build-isolation
pip install -e .
```







## Usage
## LLaVA
```
Coming soon.
```


## Qwen2.5-VL
```
Coming soon.
```




## Citation
If you find our work useful for your research, please consider citing:

```bibtex
@article{zhang2026pio,
  title={PIO-FVLM: Rethinking Training-Free Visual Token Reduction for VLM Acceleration from an Inference-Objective Perspective},
  author={Zhang, Haokui and Ou, Congyang and Yan, Dawei and Wang, Peng and Yan, Qingsen and Li, Ying and Xiao, Rong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2602.04657},
  year={2026}
}
```
