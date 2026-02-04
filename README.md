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
  <a href="https://raw.githubusercontent.com/ocy1/PIO-FVLM/main/Images/4 methods new.pdf">
    <img src="https://raw.githubusercontent.com/ocy1/PIO-FVLM/main/Images/4 methods.jpg" alt="Structure" width="100%" />
  </a>
  <br/>
  <em>
   
  </em>
</p>

<blockquote>
  <p>
    </strong> Comparison of different token selection strategies . (a) Ours; (b) Similirty with cls token and text token based; (c) cls token similarity based and enhance diversity; (d) Cls token similarity based.
  </p>
</blockquote>











## Setup

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True`.

### Weights

We provide **two-stage checkpoints**:

- **Stage I (Band-VAE)**: `models/vae.safetensors` (download: [Hugging Face](https://huggingface.co/xxfer/SALAD-Pan))
- **Stage II (Latent Diffusion)**: runs **on top of Stable Diffusion** in the Band-VAE latent space.  
  - **Stable Diffusion base**: download from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5))  
  - **Adapters**: `models/adapters.pth` (download: [Hugging Face](https://huggingface.co/xxfer/SALAD-Pan))

## Usage

### Training

We train the model in **two stages**.

- **Stage I (VAE pretraining)**

```bash
accelerate launch train_vae.py --config configs/train_vae.yaml
```

- **Stage II (Diffusion + Adapter training)**

```bash
accelerate launch train_diffusion.py --config configs/train_diffusion.yaml
```

Note: Tuning usually takes `40k~50k` steps, about `1~2` days using eight RTX 4090 GPUs in fp16. 
Reduce `batch_size` if your GPU memory is limited.

### Inference

Once the training is done, run inference:

```python
Coming soon.
```

## Results

<p align="center">
  <a href="https://salad-pan.github.io/assets/fig3.pdf">
    <img src="https://salad-pan.github.io/assets/fig3-1.png" alt="Reduced Resolution" width="100%" />
  </a>
  <br>
  <em>Visual comparison on WorldView-3 (WV-3) and QuickBird (QB) dataset at reduced resolution.</em>
  <a href="https://salad-pan.github.io/assets/fig4.pdf">
    <img src="https://salad-pan.github.io/assets/fig4-1.png" alt="Full Resolution" width="100%" />
  </a>
  <em>Visual comparison on WorldView-3 (WV-3) and QuickBird (QB) dataset at full resolution.</em>
</p>

## Citation

If you make use of our work, please cite our paper.

```bibtex
```

## Shoutouts

- Built with [🤗 Diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing !
- The interactive demo is powered by [🤗 Gradio](https://github.com/gradio-app/gradio). Thanks for open-sourcing !

