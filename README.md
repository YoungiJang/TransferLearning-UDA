# Transfer Learning — Unsupervised Domain Adaptation (UDA)

MLDL2 HW3. Unsupervised Domain Adaptation between real bird photos (CUB-200-2011)
and bird paintings (CUB-200-Paintings), under two settings:

- **C→P**: source = CUB-200 (photos), target = CUB-200-Paintings
- **P→C**: source = CUB-200-Paintings, target = CUB-200 (photos)

Train with **labeled source** + **unlabeled target**; evaluate on the target domain.
No pretrained weights, no target labels during training, same architecture for both settings.

## Setup

```bash
# 1. Create the environment (PyTorch + deps)
conda create -n uda python=3.10 -y
conda activate uda
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm info-nce-pytorch pillow gdown

# 2. Download & extract the datasets
bash download_data.sh
```

This produces the layout expected by the code:

```
CUB_200_2011/images/    # 200 classes, ~11,788 photos
CUB-200-Painting/       # 200 classes, ~3,047 paintings (smaller, class-imbalanced)
```

## Run

```bash
python DL_HW3.py
```

Trains both settings, saves checkpoints (`CtoP.pth`, `PtoC.pth`), and writes the
prediction files `predictions_CtoP.npy` and `predictions_PtoC.npy`.

> On a Slurm cluster, submit as a job instead of running on the login node
> (a `*.sbatch` script is gitignored as it is environment-specific).

## Files

| File | Purpose |
|------|---------|
| `DL_HW3.py` | Main training + evaluation script |
| `DL_HW3.ipynb` | Original starter notebook |
| `download_data.sh` | Downloads & extracts both datasets |
| `predictions_CtoP.npy` / `predictions_PtoC.npy` | Target-domain predictions (generated) |
