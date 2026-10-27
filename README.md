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
- 🏆 **36.52 dB PSNR on Vimeo90K** (Ours-B, w/o TTA).

---

## 📊 Results

Quantitative results of the **released checkpoints** with **Test-Time Augmentation (TTA) disabled**. Best results in **bold**.

> [!NOTE]
> The released checkpoints differ slightly from the configuration reported in the paper, so the numbers below may not exactly match the paper. See [Checkpoints](#-checkpoints) for the paper version.

### Standard benchmarks

| Model | Vimeo90K<br><sub>PSNR / SSIM</sub> | UCF101<br><sub>PSNR / SSIM</sub> | Xiph-2K<br><sub>PSNR / SSIM</sub> | Xiph-4K<br><sub>PSNR / SSIM</sub> | Middlebury<br><sub>IE ↓</sub> | Params<br><sub>(M)</sub> | FLOPs<br><sub>(T)</sub> |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours-C | 36.10 / 0.9801 | 35.38 / 0.9700 | 37.12 / 0.946 | 34.81 / 0.908 | 1.94 | **4.3** | **0.27** |
| Ours-E | 36.20 / 0.9802 | 35.42 / 0.9699 | 37.17 / 0.946 | 34.99 / 0.910 | 1.96 | 6.7 | 0.29 |
| Ours-B | **36.52 / 0.9810** | **35.47 / 0.9703** | **37.33 / 0.947** | **35.14 / 0.911** | **1.90** | 16.2 | 1.07 |

### SNU-FILM

| Model | Easy<br><sub>PSNR / SSIM</sub> | Medium<br><sub>PSNR / SSIM</sub> | Hard<br><sub>PSNR / SSIM</sub> | Extreme<br><sub>PSNR / SSIM</sub> |
|:--|:-:|:-:|:-:|:-:|
| Ours-C | 40.10 / **0.9915** | 36.11 / 0.9809 | 30.81 / 0.9405 | 25.69 / 0.8710 |
| Ours-E | 40.15 / 0.9912 | 36.18 / 0.9809 | 30.89 / 0.9416 | 25.81 / **0.8725** |
| Ours-B | **40.20** / 0.9909 | **36.30 / 0.9810** | **31.00 / 0.9417** | **25.83** / 0.8722 |

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

> [!IMPORTANT]
> The checkpoints in the main folder are the **released versions**, which differ slightly from the paper.
> The **original paper configuration** is available in the `deprecated` folder of the same Drive link.

| Model | Params (M) | FLOPs (T) |
|:--|:-:|:-:|
| Ours-C | 4.3 | 0.27 |
| Ours-E | 6.7 | 0.29 |
| Ours-B | 16.2 | 1.07 |

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
