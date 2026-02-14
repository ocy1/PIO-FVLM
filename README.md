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
