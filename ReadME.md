
# LC-Mamba

## Environment Setup

### Conda Environment
```bash
conda create -n LC_Mamba
conda activate LC_Mamba
```

### PyTorch Installation
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Library Installation
```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .

```

## Checkpoints
Checkpoints can be downloaded from the following link:  
[Download Checkpoints](https://www.dropbox.com/scl/fo/qqetwthteq18n53p48nwi/ALtsYM3Dse5nW5QfA5KwGEA?rlkey=lhcev43r7ltvv80cr3iszj6nj&dl=0)

Please place the checkpoints in ./ckpt/.

## Dataset
To comprehensively evaluate the proposed model's performance under various conditions and resolutions, experiments were conducted using multiple datasets.

### Dataset Structure
```
/data/datasets/
  ├── middlebury
  ├── snufilm
  ├── ucf101
  ├── vimeo_triplet
  ├── Xiph
```

### Dataset Preparation
The following datasets were used:

- **[Vimeo90K dataset](http://toflow.csail.mit.edu/):**  
  Consists of frame triplets with a resolution of 448×256. The test set includes 3,782 triplets.

- **[UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow):**  
  The test set contains 379 frame triplets selected from DVF, with a resolution of 256×256.

- **[Xiph dataset](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py):**  
  The original images were downsampled to 2K resolution as "Xiph-2K" and cropped centrally to form "Xiph-4K" for testing.

- **[Middlebury OTHER dataset](https://vision.middlebury.edu/flow/data/):**  
  The OTHER set, with a resolution of approximately 640×480, was used for testing.

- **[SNU-FILM dataset](https://myungsub.github.io/CAIN/):**  
  This dataset consists of 1,240 frame triplets with a resolution of approximately 1280×720. It is categorized into four difficulty levels—Easy, Medium, Hard, and Extreme—based on motion magnitude, enabling detailed performance comparisons.

### Benchmarks
Run the benchmark using the following command:
```bash
Make bench_Ours-CS
Make bench_Ours-ES
Make bench_Ours-BS
Make bench_Ours-PS
```

## License and Acknowledgement

This project is distributed under the Apache 2.0 license. It incorporates concepts and code from RIFE, EMA-VFI, and VFIMamba, 
and users are advised to adhere to the licensing terms of these respective projects.

We extend our gratitude to the authors of these works for their exceptional contributions.
