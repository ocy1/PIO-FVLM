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

If you make use of our work, please cite our paper.

```
```


