# Transfer Learning — Unsupervised Domain Adaptation (UDA)

Unsupervised Domain Adaptation between real bird photos (**CUB-200-2011**) and bird
paintings (**CUB-200-Paintings**), 200 fine-grained classes, under two settings:

- **C→P**: source = CUB-200 (photos), target = CUB-200-Paintings (paintings)
- **P→C**: source = CUB-200-Paintings, target = CUB-200 (photos)

The model is trained with **labeled source** + **unlabeled target** and evaluated on the
target domain. Constraints: **no pretrained weights**, **no target labels during training**,
**same architecture for both settings**.

## Method (overview)

A single `ResNet-18` trained **from scratch** (`weights=None`), refined by stacking
unsupervised domain-adaptation techniques. See [DESIGN_NOTES.md](DESIGN_NOTES.md) for the
full rationale and per-step results.

1. **Source training** — strong augmentation (RandomResizedCrop / flip / ColorJitter /
   RandAugment / RandomErasing), label smoothing, SGD + warmup&cosine LR; class-balanced
   sampler for the imbalanced Paintings source. Model selection on a held-out **source**
   split (target labels never used). Then **BatchNorm adaptation** on the unlabeled target.
2. **Pseudo-labeling** — self-train on high-confidence target predictions (joint with source).
3. **Entropy minimization** (Information Maximization) — make target predictions confident
   yet class-balanced.
4. **DANN** — domain-adversarial feature alignment via a gradient-reversal domain classifier.

## Setup

```bash
# 1. Environment (PyTorch + deps). CUDA 12.x; pick the wheel matching your driver.
conda create -n uda python=3.10 -y
conda activate uda
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm info-nce-pytorch pillow gdown mpmath

# 2. Download & extract the datasets (needs wget, unzip, gdown)
bash download_data.sh
```

`download_data.sh` produces the layout the code expects:

```
CUB_200_2011/images/    # 200 classes, ~11,788 photos
CUB-200-Painting/       # 200 classes, ~3,047 paintings (smaller, class-imbalanced)
```

## Run

Training is staged. Each stage builds on the checkpoint saved by the previous one
(`checkpoints/CtoP.pth`, `checkpoints/PtoC.pth`). The evaluation block always runs and
(re)writes the prediction files. Control via two environment variables:

- `UDA_MODE` — `source` (default) → `pseudo` → `entropy` → `dann`
- `UDA_SETTINGS` — `CtoP,PtoC` (default), or a subset to (re)train only those

```bash
# Stage 1: train ResNet-18 from scratch on both settings (+ BN adaptation)
python DL_HW3.py

# Stage 2: stack UDA refinements (each loads the previous checkpoint)
UDA_MODE=pseudo  python DL_HW3.py    # pseudo-labeling
UDA_MODE=entropy python DL_HW3.py    # entropy minimization
UDA_MODE=dann    python DL_HW3.py    # domain-adversarial (DANN)

# Retrain only one setting (keeps the other checkpoint), e.g. just P→C:
UDA_SETTINGS=PtoC python DL_HW3.py
```

Outputs:

- `checkpoints/CtoP.pth`, `checkpoints/PtoC.pth` — trained weights (plain ResNet-18 state_dict)
- `predictions_CtoP.npy`, `predictions_PtoC.npy` — target-domain predictions (submission files)

> **Running on a Slurm cluster?** A reference job script is in `slurm/run_uda.sbatch`.
> Submit with e.g. `sbatch --export=ALL,UDA_MODE=entropy slurm/run_uda.sbatch`.
> The script is cluster-specific (paths/partition) and is not required to run the code.

## Files

| Path | Purpose |
|------|---------|
| `DL_HW3.py` | Training + evaluation (all stages, selected via `UDA_MODE`) |
| `DL_HW3.ipynb` | Original starter notebook |
| `download_data.sh` | Downloads & extracts both datasets |
| `DESIGN_NOTES.md` | Method rationale + per-step experiment log |
| `checkpoints/`, `slurm/` | Local run artifacts (git-ignored) |
