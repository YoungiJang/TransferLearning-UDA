"""
DL2 Homework 3: Transfer Learning - Unsupervised Domain Adaptation (UDA)

Phase 1 improved version (replaces the trivial-CNN baseline).

Strategy (see DESIGN_NOTES.md):
  - Backbone : ResNet-18 from scratch (weights=None) -- same for both settings.
  - Source   : strong supervised training (RandomResizedCrop/flip/ColorJitter/
               RandAugment/RandomErasing) + label smoothing + SGD + warmup&cosine LR.
  - Imbalance: class-balanced sampler when the source is CUB-200-Paintings (P->C).
  - Selection: 90/10 source train/val split; pick the best epoch by SOURCE-val
               accuracy (target labels are never used).
  - UDA      : BatchNorm adaptation -- recompute BN running stats on the (unlabeled)
               target domain before saving the checkpoint.

Constraints respected:
  - No pretrained weights (resnet18(weights=None) == random init).
  - No target labels during training / model selection.
  - Same architecture for both settings.
  - The [Evaluation and Submit] section logic is kept identical (only `model` is a
    ResNet-18 instead of the starter CNN).

Settings:
  [Setting1] CtoP : Source = CUB-200,          Target = CUB-200-Paintings
  [Setting2] PtoC : Source = CUB-200-Paintings, Target = CUB-200
"""

import os
import math
import copy
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Paths (local; datasets fetched via download_data.sh)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = BASE_DIR          # predictions_*.npy (submission artifacts) stay at root
os.makedirs(output_dir, exist_ok=True)

# Model checkpoints live in a dedicated folder.
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

CUB_200_PATH = os.path.join(BASE_DIR, "CUB_200_2011", "images")
CUB_200_PAINTINGS_PATH = os.path.join(BASE_DIR, "CUB-200-Painting")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 200

# Domain-specific normalization stats (must match the fixed evaluation section).
CUB_MEAN, CUB_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PNT_MEAN, PNT_STD = (0.7815, 0.7699, 0.7322), (0.2654, 0.2694, 0.2941)


# -----------------------------------------------------------------------------
# Model (same architecture for both settings; no pretrained weights)
# -----------------------------------------------------------------------------
def build_model():
    return resnet18(weights=None, num_classes=NUM_CLASSES)


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def train_transform(mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])


def eval_transform(mean, std):
    # Deterministic; matches the fixed evaluation transform.
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# Per-setting domains: (source_root, source_stats, target_root, target_stats)
DOMAINS = {
    "CtoP": dict(src_root=CUB_200_PATH, src_stats=(CUB_MEAN, CUB_STD),
                 tgt_root=CUB_200_PAINTINGS_PATH, tgt_stats=(PNT_MEAN, PNT_STD)),
    "PtoC": dict(src_root=CUB_200_PAINTINGS_PATH, src_stats=(PNT_MEAN, PNT_STD),
                 tgt_root=CUB_200_PATH, tgt_stats=(CUB_MEAN, CUB_STD)),
}

# Per-setting training config (architecture identical; hyperparameters may differ).
CONFIGS = {
    # CUB source: abundant & balanced -> plain shuffling, higher LR.
    "CtoP": dict(epochs=80, warmup=5, lr=0.1, batch=128, balanced=False),
    # Paintings source: small & imbalanced -> balanced sampler, gentler LR.
    # [Log 2] epochs 80->180 (Phase 1 showed P->C still underfit at ep80).
    "PtoC": dict(epochs=180, warmup=10, lr=0.05, batch=128, balanced=True),
}


# -----------------------------------------------------------------------------
# Data loaders for one setting
# -----------------------------------------------------------------------------
def build_loaders(setting, batch_size, balanced):
    dom = DOMAINS[setting]
    src_mean, src_std = dom["src_stats"]
    tgt_mean, tgt_std = dom["tgt_stats"]

    # Two views of the same source folder: train (augmented) and eval (deterministic).
    src_train_full = ImageFolder(dom["src_root"], transform=train_transform(src_mean, src_std))
    src_eval_full = ImageFolder(dom["src_root"], transform=eval_transform(src_mean, src_std))

    # 90/10 stratified-ish split via a seeded permutation (source labels allowed).
    n = len(src_train_full)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(0.1 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = Subset(src_train_full, train_idx)
    val_set = Subset(src_eval_full, val_idx)

    if balanced:
        targets = src_train_full.targets
        train_targets = [targets[i] for i in train_idx]
        class_count = np.bincount(train_targets, minlength=NUM_CLASSES)
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = [class_weight[t] for t in train_targets]
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(train_idx),
                                        replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  num_workers=8, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    # Unlabeled target (for BN adaptation); deterministic target-domain transform.
    target_set = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=True,
                               num_workers=8, pin_memory=True)
    return train_loader, val_loader, target_loader


# -----------------------------------------------------------------------------
# Train / validate
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        outputs = model(images.to(device))
        pred = outputs.argmax(dim=1).cpu()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def adapt_bn(model, target_loader):
    """BatchNorm adaptation: recompute BN running stats on the unlabeled target."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over the target pass
    model.train()
    with torch.no_grad():
        for images, _ in tqdm(target_loader, desc="BN-adapt"):
            model(images.to(device))
    model.eval()


def train_one_setting(setting):
    cfg = CONFIGS[setting]
    print(f"\n========== Training [{setting}] ==========")
    print(f"config: {cfg}")
    train_loader, val_loader, target_loader = build_loaders(
        setting, cfg["batch"], cfg["balanced"])

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                                nesterov=True, weight_decay=5e-4)

    warmup, epochs = cfg["warmup"], cfg["epochs"]

    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        loss_total, n_iter = 0.0, 0
        for images, targets in tqdm(train_loader, desc=f"[{setting}] ep{epoch+1}/{epochs}"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            n_iter += 1
        scheduler.step()

        val_acc = evaluate(model, val_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{setting}] epoch {epoch+1}/{epochs} - loss {loss_total/n_iter:.3f} "
              f"- src_val_acc {val_acc:.2f}% - lr {lr_now:.4f}")

        # Model selection on SOURCE validation only (no target labels).
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    print(f"[{setting}] best source-val acc: {best_val:.2f}%")

    # Restore best, then adapt BN to the unlabeled target domain.
    model.load_state_dict(best_state)
    print(f"[{setting}] BatchNorm adaptation on unlabeled target...")
    adapt_bn(model, target_loader)
    return model


# -----------------------------------------------------------------------------
# Phase 2-(1): Pseudo-labeling self-training  ([Log 4])
#   Build on a trained checkpoint: keep target predictions with confidence >= tau
#   as pseudo-labels, fine-tune jointly with the labeled source for a few rounds.
# -----------------------------------------------------------------------------
PSEUDO_CONFIGS = {
    "CtoP": dict(tau=0.9, rounds=3, ft_epochs=10, ft_lr=0.01, batch=128),
    "PtoC": dict(tau=0.9, rounds=3, ft_epochs=10, ft_lr=0.01, batch=128),
}


class PseudoLabeledSubset(Dataset):
    """Selected target images (with strong aug) paired with their pseudo-labels."""
    def __init__(self, base_dataset, indices, labels):
        self.base = base_dataset
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return img, int(self.labels[i])


@torch.no_grad()
def generate_pseudo_labels(model, target_eval_loader, tau):
    model.eval()
    confs, preds = [], []
    for images, _ in target_eval_loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        c, p = probs.max(dim=1)
        confs.append(c.cpu())
        preds.append(p.cpu())
    conf = torch.cat(confs)
    pred = torch.cat(preds)
    mask = conf >= tau
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
    labels = pred[mask].tolist()
    return idx, labels, conf, pred


def pseudo_finetune(setting, init_ckpt):
    cfg = CONFIGS[setting]
    pcfg = PSEUDO_CONFIGS[setting]
    dom = DOMAINS[setting]
    tgt_mean, tgt_std = dom["tgt_stats"]
    bs = pcfg["batch"]
    print(f"\n========== Pseudo-labeling [{setting}] ==========")
    print(f"pseudo config: {pcfg}")

    # Labeled source (strong aug) + source-val for monitoring.
    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])

    # Target: deterministic view (for label generation) + augmented view (for fine-tune).
    target_eval = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    target_aug = ImageFolder(dom["tgt_root"], transform=train_transform(tgt_mean, tgt_std))
    target_eval_loader = DataLoader(target_eval, batch_size=bs, shuffle=False,
                                    num_workers=8, pin_memory=True)
    n_target = len(target_eval)

    model = build_model().to(device)
    model.load_state_dict(torch.load(init_ckpt))
    print(f"[{setting}] loaded init checkpoint: {init_ckpt}")
    print(f"[{setting}] init source-val: {evaluate(model, val_loader):.2f}%")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for r in range(pcfg["rounds"]):
        idx, plabels, conf, pred = generate_pseudo_labels(model, target_eval_loader, pcfg["tau"])
        n_sel, n_cls = len(idx), len(set(plabels))
        print(f"[{setting}] round {r+1}/{pcfg['rounds']}: pseudo-labeled "
              f"{n_sel}/{n_target} ({100*n_sel/max(n_target,1):.1f}%), "
              f"classes covered {n_cls}/{NUM_CLASSES}, mean-conf {conf.mean():.3f}")
        if n_sel < bs:
            print("  too few confident samples; stopping pseudo rounds.")
            break

        # Iterate over the (larger) SOURCE loader each epoch; cycle the pseudo set
        # so the few confident target samples are reused -- otherwise a tiny pseudo
        # set would give only ~1 iteration/epoch. drop_last=False to use every sample.
        pseudo_loader = DataLoader(PseudoLabeledSubset(target_aug, idx, plabels),
                                   batch_size=bs, shuffle=True, num_workers=8,
                                   pin_memory=True, drop_last=False)

        optimizer = torch.optim.SGD(model.parameters(), lr=pcfg["ft_lr"], momentum=0.9,
                                    nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pcfg["ft_epochs"])
        pseudo_iter = cycle(pseudo_loader)

        for epoch in range(pcfg["ft_epochs"]):
            model.train()
            loss_t_tot, n_iter = 0.0, 0
            for images_s, labels_s in tqdm(source_loader,
                                           desc=f"[{setting}] r{r+1} ft{epoch+1}/{pcfg['ft_epochs']}"):
                images_s, labels_s = images_s.to(device), labels_s.to(device)
                images_t, labels_t = next(pseudo_iter)
                images_t, labels_t = images_t.to(device), labels_t.to(device)

                optimizer.zero_grad()
                loss_s = criterion(model(images_s), labels_s)   # source anchor
                loss_t = criterion(model(images_t), labels_t)   # pseudo-labeled target
                loss = loss_s + loss_t
                loss.backward()
                optimizer.step()
                loss_t_tot += loss_t.item()
                n_iter += 1
            scheduler.step()
            val_acc = evaluate(model, val_loader)
            print(f"[{setting}] r{r+1} ft{epoch+1}/{pcfg['ft_epochs']} - "
                  f"pseudo_loss {loss_t_tot/max(n_iter,1):.3f} - src_val_acc {val_acc:.2f}%")

    print(f"[{setting}] BatchNorm adaptation on unlabeled target...")
    adapt_bn(model, target_eval_loader)
    return model


# -----------------------------------------------------------------------------
# Phase 2-(2): Entropy minimization / Information Maximization  ([Log 5])
#   No pseudo-labels: directly shape target predictions to be confident
#   (low per-sample entropy) yet class-balanced (high entropy of the mean).
# -----------------------------------------------------------------------------
ENTROPY_CONFIGS = {
    "CtoP": dict(lam=1.0, ft_epochs=15, ft_lr=0.01, batch=128),
    "PtoC": dict(lam=1.0, ft_epochs=15, ft_lr=0.01, batch=128),
}


def information_maximization_loss(logits):
    """L_IM = mean_i H(p_i) - H(p_bar): confident per-sample + balanced overall."""
    p = torch.softmax(logits, dim=1)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()       # minimize -> confident
    p_bar = p.mean(dim=0)
    div = -(p_bar * torch.log(p_bar + 1e-8)).sum()           # H(mean); maximize
    return ent - div


def entropy_finetune(setting, init_ckpt):
    cfg = CONFIGS[setting]
    ecfg = ENTROPY_CONFIGS[setting]
    dom = DOMAINS[setting]
    tgt_mean, tgt_std = dom["tgt_stats"]
    bs = ecfg["batch"]
    print(f"\n========== Entropy-min [{setting}] ==========")
    print(f"entropy config: {ecfg}")

    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])

    target_set = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    # drop_last=True so the diversity term H(p_bar) is estimated on full batches.
    target_loader = DataLoader(target_set, batch_size=bs, shuffle=True, num_workers=8,
                               pin_memory=True, drop_last=True)
    target_bn_loader = DataLoader(target_set, batch_size=bs, shuffle=False,
                                  num_workers=8, pin_memory=True)

    model = build_model().to(device)
    model.load_state_dict(torch.load(init_ckpt))
    print(f"[{setting}] loaded init checkpoint: {init_ckpt}")
    print(f"[{setting}] init source-val: {evaluate(model, val_loader):.2f}%")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=ecfg["ft_lr"], momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ecfg["ft_epochs"])
    target_iter = cycle(target_loader)

    for epoch in range(ecfg["ft_epochs"]):
        model.train()
        im_tot, n_iter = 0.0, 0
        for images_s, labels_s in tqdm(source_loader,
                                       desc=f"[{setting}] ent ep{epoch+1}/{ecfg['ft_epochs']}"):
            images_s, labels_s = images_s.to(device), labels_s.to(device)
            images_t, _ = next(target_iter)
            images_t = images_t.to(device)

            optimizer.zero_grad()
            loss_s = criterion(model(images_s), labels_s)          # source anchor
            loss_im = information_maximization_loss(model(images_t))  # target IM
            loss = loss_s + ecfg["lam"] * loss_im
            loss.backward()
            optimizer.step()
            im_tot += loss_im.item()
            n_iter += 1
        scheduler.step()
        val_acc = evaluate(model, val_loader)
        print(f"[{setting}] ent ep{epoch+1}/{ecfg['ft_epochs']} - "
              f"IM_loss {im_tot/max(n_iter,1):.3f} - src_val_acc {val_acc:.2f}%")

    print(f"[{setting}] BatchNorm adaptation on unlabeled target...")
    adapt_bn(model, target_bn_loader)
    return model


# -----------------------------------------------------------------------------
# Phase 2-(3): DANN -- Domain-Adversarial training  ([Log 6])
#   Align the 512-d features of source/target via a gradient-reversal domain
#   classifier, stacked on top of source CE and target IM.
# -----------------------------------------------------------------------------
DANN_CONFIGS = {
    "CtoP": dict(epochs=15, lr=0.01, batch=128, lam_im=1.0),
    "PtoC": dict(epochs=15, lr=0.01, batch=128, lam_im=1.0),
}


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DANN(nn.Module):
    """Wrap a resnet18: shared features -> label head (resnet.fc) + domain head."""
    def __init__(self, resnet, feat_dim=512):
        super().__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.classifier = resnet.fc                                   # 512 -> 200
        self.domain = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x, lambd=0.0):
        f = self.features(x).flatten(1)
        return self.classifier(f), self.domain(grad_reverse(f, lambd))


def dann_finetune(setting, init_ckpt):
    cfg = CONFIGS[setting]
    dcfg = DANN_CONFIGS[setting]
    dom = DOMAINS[setting]
    tgt_mean, tgt_std = dom["tgt_stats"]
    bs = dcfg["batch"]
    print(f"\n========== DANN [{setting}] ==========")
    print(f"dann config: {dcfg}")

    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])
    target_set = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    target_loader = DataLoader(target_set, batch_size=bs, shuffle=True, num_workers=8,
                               pin_memory=True, drop_last=True)
    target_bn_loader = DataLoader(target_set, batch_size=bs, shuffle=False,
                                  num_workers=8, pin_memory=True)

    resnet = build_model().to(device)
    resnet.load_state_dict(torch.load(init_ckpt))
    model = DANN(resnet).to(device)
    print(f"[{setting}] loaded init checkpoint: {init_ckpt}")
    print(f"[{setting}] init source-val: {evaluate(resnet, val_loader):.2f}%")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=dcfg["lr"], momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, dcfg["epochs"])
    target_iter = cycle(target_loader)

    total_steps = dcfg["epochs"] * len(source_loader)
    step = 0
    for epoch in range(dcfg["epochs"]):
        model.train()
        dom_tot, n_iter = 0.0, 0
        for images_s, labels_s in tqdm(source_loader,
                                       desc=f"[{setting}] dann ep{epoch+1}/{dcfg['epochs']}"):
            images_s, labels_s = images_s.to(device), labels_s.to(device)
            images_t, _ = next(target_iter)
            images_t = images_t.to(device)

            p = step / max(total_steps, 1)
            lambd = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0   # GRL strength 0 -> 1

            optimizer.zero_grad()
            out_s, dom_s = model(images_s, lambd)
            out_t, dom_t = model(images_t, lambd)
            d_s = torch.zeros(images_s.size(0), dtype=torch.long, device=device)
            d_t = torch.ones(images_t.size(0), dtype=torch.long, device=device)

            loss_cls = criterion(out_s, labels_s)                      # source labels
            loss_dom = criterion(dom_s, d_s) + criterion(dom_t, d_t)   # adversarial (via GRL)
            loss_im = information_maximization_loss(out_t)             # target IM
            loss = loss_cls + loss_dom + dcfg["lam_im"] * loss_im
            loss.backward()
            optimizer.step()
            dom_tot += loss_dom.item()
            n_iter += 1
            step += 1
        scheduler.step()
        val_acc = evaluate(resnet, val_loader)
        print(f"[{setting}] dann ep{epoch+1}/{dcfg['epochs']} - "
              f"dom_loss {dom_tot/max(n_iter,1):.3f} - lambda {lambd:.2f} - "
              f"src_val_acc {val_acc:.2f}%")

    print(f"[{setting}] BatchNorm adaptation on unlabeled target...")
    adapt_bn(resnet, target_bn_loader)
    return resnet


# -----------------------------------------------------------------------------
# Run training for both settings and save checkpoints.
# -----------------------------------------------------------------------------
CtoP_CKPT_PATH = os.path.join(CKPT_DIR, "CtoP.pth")
PtoC_CKPT_PATH = os.path.join(CKPT_DIR, "PtoC.pth")
BATCH_SIZE = 128

CKPT_PATHS = {"CtoP": CtoP_CKPT_PATH, "PtoC": PtoC_CKPT_PATH}

if __name__ == "__main__":
    # Which settings to (re)train this run. Default: both.
    # Override to save compute, e.g. UDA_SETTINGS=PtoC to retrain only P->C and
    # keep the existing CtoP.pth. Evaluation below always runs both settings.
    settings_to_train = [s.strip() for s in
                         os.environ.get("UDA_SETTINGS", "CtoP,PtoC").split(",")
                         if s.strip()]
    # UDA_MODE: "source" = Phase 1 from-scratch training; "pseudo" = Phase 2-(1)
    # pseudo-labeling self-training that builds on the existing checkpoint.
    mode = os.environ.get("UDA_MODE", "source")
    print(f"Mode: {mode} | settings: {settings_to_train}")
    for s in settings_to_train:
        if mode == "pseudo":
            model = pseudo_finetune(s, CKPT_PATHS[s])
        elif mode == "entropy":
            model = entropy_finetune(s, CKPT_PATHS[s])
        elif mode == "dann":
            model = dann_finetune(s, CKPT_PATHS[s])
        else:
            model = train_one_setting(s)
        torch.save(model.state_dict(), CKPT_PATHS[s])
        print(f"[{s}] saved checkpoint -> {CKPT_PATHS[s]}")

    # =========================================================================
    # 5. Evaluation and Submit
    #   [Setting1] CtoP : Evaluate on CUB-200-Paintings.
    #   [Setting2] PtoC : Evaluate on CUB-200.
    # Logic kept identical to the starter "DO NOT CHANGE" cells. `model` is a
    # ResNet-18, and is put in eval mode so the target-adapted BN running stats
    # are used at inference.
    # =========================================================================
    model = build_model().to(device)
    model.eval()

    # ---- [Evaluation: Setting1] Source: CUB-200, Target: CUB-200-Paintings ----
    # ======================== DO NOT CHANGE THE LOGIC BELOW ======================== #
    model.load_state_dict(torch.load(CtoP_CKPT_PATH))

    cub_bird_paintings_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.7815, 0.7699, 0.7322), (0.2654, 0.2694, 0.2941)),
    ])
    cub_paintings_dataset = ImageFolder(root=CUB_200_PAINTINGS_PATH,
                                        transform=cub_bird_paintings_transform)
    val_dataset = cub_paintings_dataset
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

    correct, total, predictions_list = 0, 0, []
    with torch.no_grad():
        for data in tqdm(val_data_loader):
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            predictions_list.extend(predicted.cpu().numpy())

    predictions_array = np.array(predictions_list)
    np.save(f"{output_dir}/predictions_CtoP.npy", predictions_array)
    print(f"[Evaluation1] Accuracy of the network on the CUB-200-Paintings: "
          f"{100 * correct / total:.2f} %")
    # ============================================================================== #

    # ---- [Evaluation: Setting2] Source: CUB-200-Paintings, Target: CUB-200 ----
    # ======================== DO NOT CHANGE THE LOGIC BELOW ======================== #
    model.load_state_dict(torch.load(PtoC_CKPT_PATH))

    cub_bird_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    cub_dataset = ImageFolder(root=CUB_200_PATH, transform=cub_bird_transform)
    val_dataset = cub_dataset
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

    correct, total, predictions_list = 0, 0, []
    with torch.no_grad():
        for data in tqdm(val_data_loader):
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            predictions_list.extend(predicted.cpu().numpy())

    predictions_array = np.array(predictions_list)
    np.save(f"{output_dir}/predictions_PtoC.npy", predictions_array)
    print(f"[Evaluation2] Accuracy of the network on the CUB-200: "
          f"{100 * correct / total:.2f} %")
    # ============================================================================== #
