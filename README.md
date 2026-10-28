<div align="center">

# 🐍 LC-Mamba

### Local and Continuous Mamba with Shifted Windows for Frame Interpolation

**Min Wu Jeong** · **Chae Eun Rhee**

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-4b44ce.svg)](https://cvpr.thecvf.com/virtual/2025/poster/34432)
[![Paper](https://img.shields.io/badge/Paper-CVF%20Open%20Access-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_LC-Mamba_Local_and_Continuous_Mamba_with_Shifted_Windows_for_Frame_CVPR_2025_paper.html)
[![Supplementary](https://img.shields.io/badge/Supp-PDF-orange.svg)](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Jeong_LC-Mamba_Local_and_CVPR_2025_supplemental.pdf)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Google%20Drive-34a853.svg)](https://drive.google.com/drive/folders/1nmnWW-73MtSSJmnfBRTbvBjMsjNSG3To?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#-license--acknowledgement)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## ✨ Highlights

LC-Mamba is a Mamba-based **video frame interpolation (VFI)** model designed to capture fine-grained spatiotemporal detail.

- 🪟 **Shifted local windows** — the selective scan runs inside local windows, reducing the long-range decay of historical information and strengthening multi-scale local features.
- 🌀 **Hilbert-curve selective state scan** — tokens are ordered along a Hilbert curve so spatial continuity is preserved within *and* across window boundaries.
- 🧊 **Voxel-level Hilbert scanning** — the curve is extended to 3D to model spatiotemporal patterns between frames.
- 🏆 **36.53 dB PSNR on Vimeo90K** (paper, Ours-P) · **36.52 dB** with only 16.2M params (improved Ours-B), w/o TTA.

---

## 📊 Results

All results are reported with **Test-Time Augmentation (TTA) disabled**. Best results among our models in **bold**.

LC-Mamba has two sets of checkpoints:

| Version | Models | Results | Checkpoints |
|:--|:-:|:--|:--|
| **Improved** (recommended) | Ours-C / E / B | Supplementary Table 8 | Drive main folder |
| **Paper** (for reproducing Table 1) | Ours-E / B / P | Main paper Table 1 | Drive `deprecated` folder |

### Improved checkpoints (recommended)

| Model | Vimeo90K<br><sub>PSNR / SSIM</sub> | UCF101<br><sub>PSNR / SSIM</sub> | Xiph-2K<br><sub>PSNR / SSIM</sub> | Xiph-4K<br><sub>PSNR / SSIM</sub> | Middlebury<br><sub>IE ↓</sub> | Params<br><sub>(M)</sub> | FLOPs<br><sub>(T)</sub> |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours-C | 36.10 / 0.9801 | 35.38 / 0.9700 | 37.12 / 0.946 | 34.81 / 0.908 | 1.94 | **4.3** | **0.27** |
| Ours-E | 36.20 / 0.9802 | 35.42 / 0.9699 | 37.17 / 0.946 | 34.99 / 0.910 | 1.96 | 6.7 | 0.29 |
| Ours-B | **36.52 / 0.9810** | **35.47 / 0.9703** | **37.33 / 0.947** | **35.14 / 0.911** | **1.90** | 16.2 | 1.07 |

| Model | SNU-FILM Easy<br><sub>PSNR / SSIM</sub> | Medium<br><sub>PSNR / SSIM</sub> | Hard<br><sub>PSNR / SSIM</sub> | Extreme<br><sub>PSNR / SSIM</sub> |
|:--|:-:|:-:|:-:|:-:|
| Ours-C | 40.10 / **0.9915** | 36.11 / 0.9809 | 30.81 / 0.9405 | 25.69 / 0.8710 |
| Ours-E | 40.15 / 0.9912 | 36.18 / 0.9809 | 30.89 / 0.9416 | 25.81 / **0.8725** |
| Ours-B | **40.20** / 0.9909 | **36.30 / 0.9810** | **31.00 / 0.9417** | **25.83** / 0.8722 |

<details>
<summary><b>Paper checkpoints (main paper Table 1)</b></summary>

<br>

| Model | Vimeo90K<br><sub>PSNR / SSIM</sub> | UCF101<br><sub>PSNR / SSIM</sub> | Xiph-2K<br><sub>PSNR / SSIM</sub> | Xiph-4K<br><sub>PSNR / SSIM</sub> | Middlebury<br><sub>IE ↓</sub> | Params<br><sub>(M)</sub> |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours-E | 36.19 / 0.9803 | 35.33 / 0.9695 | 36.67 / 0.943 | 34.26 / 0.903 | 1.98 | **6.7** |
| Ours-B | 36.43 / 0.9813 | 35.39 / 0.9698 | 36.90 / 0.945 | 34.26 / 0.904 | **1.89** | 16.2 |
| Ours-P | **36.53 / 0.9816** | **35.42 / 0.9699** | **36.99 / 0.946** | **34.49 / 0.906** | 1.92 | 25.4 |

| Model | SNU-FILM Easy<br><sub>PSNR / SSIM</sub> | Medium<br><sub>PSNR / SSIM</sub> | Hard<br><sub>PSNR / SSIM</sub> | Extreme<br><sub>PSNR / SSIM</sub> |
|:--|:-:|:-:|:-:|:-:|
| Ours-E | 39.82 / 0.9907 | 35.87 / 0.9797 | 30.54 / 0.9373 | 25.33 / 0.8626 |
| Ours-B | 40.07 / **0.9909** | 36.08 / 0.9801 | 30.59 / 0.9375 | 25.35 / 0.8630 |
| Ours-P | **40.16 / 0.9909** | **36.17 / 0.9802** | **30.72 / 0.9382** | **25.48 / 0.8645** |

</details>

---

## 🛠️ Installation

```bash
# 1. Conda environment
conda create -n LC_Mamba python=3.10 -y
conda activate LC_Mamba

# 2. PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Dependencies & selective-scan kernel
pip install -r requirements.txt
cd kernels/selective_scan && pip install . && cd ../..
```

---

## 📦 Checkpoints

Download the pretrained weights from **[Google Drive](https://drive.google.com/drive/folders/1nmnWW-73MtSSJmnfBRTbvBjMsjNSG3To?usp=sharing)** and place them under `./ckpt/`.

- **Main folder** — improved checkpoints (**Ours-C / E / B**). Recommended.
- **`deprecated` folder** — original paper checkpoints (**Ours-E / B / P**), for reproducing main paper Table 1.

> [!NOTE]
> Ours-E and Ours-B share names across both versions but are **different weights**. Check which version you load when comparing against the paper.

---

## 📁 Datasets

Organize the datasets as follows (default root: `/data/datasets`, change with `--datasets_path`):

```
/data/datasets/
├── middlebury
├── snufilm
├── ucf101
├── vimeo_triplet
└── Xiph
```

| Dataset | Resolution | Test set |
|:--|:-:|:--|
| [Vimeo90K](http://toflow.csail.mit.edu/) | 448×256 | 3,782 triplets |
| [UCF101](https://liuziwei7.github.io/projects/VoxelFlow) | 256×256 | 379 triplets (from DVF) |
| [Xiph](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py) | 2K / 4K | Downsampled to 2K (Xiph-2K), center-cropped (Xiph-4K) |
| [Middlebury OTHER](https://vision.middlebury.edu/flow/data/) | ~640×480 | OTHER set |
| [SNU-FILM](https://myungsub.github.io/CAIN/) | ~1280×720 | 1,240 triplets · Easy / Medium / Hard / Extreme |

---

## 🚀 Benchmark

```bash
make bench_Ours-C
make bench_Ours-E
make bench_Ours-B
```

Or run directly with a custom dataset path:

```bash
python model_benchmark.py --model Ours-B --resume Ours-B --datasets_path /path/to/datasets
```

Logs are written to `./benchlog/`.

Shifted windows default to `shift_mode="padding"`. To use cyclic shifts with
independent SSM states for boundary regions, set the backbone option before
creating the model:

```python
model_config = Model_create('Ours-C')
model_config['MODEL_ARCH'][0]['shift_mode'] = 'roll'
model = Model(-1, model_config)
```

The two modes use the same weights, Hilbert token order, and convolution boundary
conditions for either `conv_first` setting. The roll mode packs each region into
an independent scan and omits padding tokens, so outputs and gradients match the
padding mode within floating-point tolerance. Scan indices are cached by input
size; the extra gathering and separate scan calls can affect runtime.

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{jeong2025lcmamba,
  title     = {LC-Mamba: Local and Continuous Mamba with Shifted Windows for Frame Interpolation},
  author    = {Jeong, Min Wu and Rhee, Chae Eun},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {17671--17681},
  year      = {2025}
}
```

---

## 🙏 License & Acknowledgement

This project is released under the **Apache 2.0** license.
It builds on concepts and code from [RIFE](https://github.com/hzwer/ECCV2022-RIFE), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI), and [VFIMamba](https://github.com/MCG-NJU/VFIMamba) — please also follow their respective licenses.
We sincerely thank the authors for their excellent work.
