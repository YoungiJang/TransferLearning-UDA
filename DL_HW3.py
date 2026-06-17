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
from torchvision.models import resnet34, resnet50
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Paths / device
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "predictions")     # prediction .npy files
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")        # model checkpoints
os.makedirs(output_dir, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CUB_200_PATH = os.path.join(BASE_DIR, "CUB_200_2011", "images")
CUB_200_PAINTINGS_PATH = os.path.join(BASE_DIR, "CUB-200-Painting")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 200

# Domain-specific normalization stats.
CUB_MEAN, CUB_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PNT_MEAN, PNT_STD = (0.7815, 0.7699, 0.7322), (0.2654, 0.2694, 0.2941)

# Per-setting domains: source/target roots and normalization stats.
DOMAINS = {
    "CtoP": dict(src_root=CUB_200_PATH, src_stats=(CUB_MEAN, CUB_STD),
                 tgt_root=CUB_200_PAINTINGS_PATH, tgt_stats=(PNT_MEAN, PNT_STD)),
    "PtoC": dict(src_root=CUB_200_PAINTINGS_PATH, src_stats=(PNT_MEAN, PNT_STD),
                 tgt_root=CUB_200_PATH, tgt_stats=(CUB_MEAN, CUB_STD)),
}

# Per-setting source-training config (architecture identical across settings).
CONFIGS = {
    # CUB source: abundant & balanced -> plain shuffling, higher LR.
    "CtoP": dict(epochs=120, warmup=5, lr=0.1, batch=128, balanced=False, mixup_alpha=0.2),
    # Paintings source: small & imbalanced -> balanced sampler, gentler LR, more epochs.
    "PtoC": dict(epochs=200, warmup=10, lr=0.05, batch=128, balanced=True, mixup_alpha=0.2),
}

CtoP_CKPT_PATH = os.path.join(CKPT_DIR, "CtoP.pth")
PtoC_CKPT_PATH = os.path.join(CKPT_DIR, "PtoC.pth")
CKPT_PATHS = {"CtoP": CtoP_CKPT_PATH, "PtoC": PtoC_CKPT_PATH}
BATCH_SIZE = 128


# Model  (ResNet-34 student)
def build_model():
    return resnet34(weights=None, num_classes=NUM_CLASSES)


def build_backbone(name):
    if name == "resnet50":
        return resnet50(weights=None, num_classes=NUM_CLASSES)
    return resnet34(weights=None, num_classes=NUM_CLASSES)


FEAT_DIM = {"resnet34": 512, "resnet50": 2048}


# Transforms
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


# Data loaders for one setting (seed varies for ensemble diversity)
def build_loaders(setting, batch_size, balanced, seed=42):
    dom = DOMAINS[setting]
    src_mean, src_std = dom["src_stats"]
    tgt_mean, tgt_std = dom["tgt_stats"]

    src_train_full = ImageFolder(dom["src_root"], transform=train_transform(src_mean, src_std))
    src_eval_full = ImageFolder(dom["src_root"], transform=eval_transform(src_mean, src_std))

    # 90/10 source train/val split (source labels allowed); seeded permutation.
    n = len(src_train_full)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, int(0.1 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_set, val_set = Subset(src_train_full, train_idx), Subset(src_eval_full, val_idx)

    if balanced:
        targets = src_train_full.targets
        train_targets = [targets[i] for i in train_idx]
        class_count = np.bincount(train_targets, minlength=NUM_CLASSES)
        sample_weight = [1.0 / max(class_count[t], 1) for t in train_targets]
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  num_workers=8, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    target_set = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=True,
                               num_workers=8, pin_memory=True)
    return train_loader, val_loader, target_loader


# Common utilities
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        pred = model(images.to(device)).argmax(dim=1).cpu()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def eval_target(model, setting):
    dom = DOMAINS[setting]
    tgt_mean, tgt_std = dom["tgt_stats"]
    ds = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    return evaluate(model, loader)


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


def target_loaders(setting, batch_size):
    """Deterministic (label-gen / BN) and shuffled-aug target loaders."""
    dom = DOMAINS[setting]
    tgt_mean, tgt_std = dom["tgt_stats"]
    eval_set = ImageFolder(dom["tgt_root"], transform=eval_transform(tgt_mean, tgt_std))
    aug_set = ImageFolder(dom["tgt_root"], transform=train_transform(tgt_mean, tgt_std))
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)
    return eval_set, aug_set, eval_loader


class PseudoLabeledSubset(Dataset):
    """Selected target images (with strong aug) paired with their pseudo-labels."""
    def __init__(self, base_dataset, indices, labels):
        self.base, self.indices, self.labels = base_dataset, indices, labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return img, int(self.labels[i])


# 1. Source training (ResNet from scratch, MixUp, source-val selection, BN-adapt)
def train_source(setting, seed=42, arch="resnet34"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = CONFIGS[setting]
    print(f"\n[train_source] setting={setting} seed={seed} arch={arch} cfg={cfg}")
    train_loader, val_loader, target_loader = build_loaders(
        setting, cfg["batch"], cfg["balanced"], seed=seed)

    model = build_backbone(arch).to(device)
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
    mixup_alpha = cfg["mixup_alpha"]
    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"[{setting}] src ep{epoch+1}/{epochs}"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[perm]
                out = model(images)
                loss = lam * criterion(out, targets) + (1 - lam) * criterion(out, targets[perm])
            else:
                loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate(model, val_loader)   # SOURCE val only (no target labels)
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
    print(f"[{setting}] best source-val acc: {best_val:.2f}%")

    model.load_state_dict(best_state)
    adapt_bn(model, target_loader)
    return model


# 2a. Entropy minimization / Information Maximization (target, label-free)
def information_maximization_loss(logits):
    """L_IM = mean_i H(p_i) - H(p_bar): confident per-sample + class-balanced overall."""
    p = torch.softmax(logits, dim=1)
    ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
    p_bar = p.mean(dim=0)
    div = -(p_bar * torch.log(p_bar + 1e-8)).sum()
    return ent - div


def entropy_refine(model, setting, epochs=15, lr=0.01, lam=1.0):
    cfg = CONFIGS[setting]
    bs = cfg["batch"]
    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])
    _, _, tgt_eval_loader = target_loaders(setting, bs)
    tgt_train_loader = DataLoader(tgt_eval_loader.dataset, batch_size=bs, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    tgt_iter = cycle(tgt_train_loader)
    for epoch in range(epochs):
        model.train()
        for images_s, labels_s in tqdm(source_loader, desc=f"[{setting}] IM ep{epoch+1}/{epochs}"):
            images_s, labels_s = images_s.to(device), labels_s.to(device)
            images_t, _ = next(tgt_iter)
            optimizer.zero_grad()
            loss = criterion(model(images_s), labels_s) + \
                lam * information_maximization_loss(model(images_t.to(device)))
            loss.backward()
            optimizer.step()
        scheduler.step()
    adapt_bn(model, tgt_eval_loader)
    return model


# 2b. DANN domain-adversarial feature alignment
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
    """Wrap a resnet: shared features -> label head (resnet.fc) + domain head (via GRL)."""
    def __init__(self, resnet, feat_dim=512):
        super().__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = resnet.fc
        self.domain = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                                    nn.Dropout(0.5), nn.Linear(256, 2))

    def forward(self, x, lambd=0.0):
        f = self.features(x).flatten(1)
        return self.classifier(f), self.domain(grad_reverse(f, lambd))


def dann_refine(model, setting, epochs=15, lr=0.01, lam_im=1.0):
    cfg = CONFIGS[setting]
    bs = cfg["batch"]
    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])
    _, _, tgt_eval_loader = target_loaders(setting, bs)
    tgt_train_loader = DataLoader(tgt_eval_loader.dataset, batch_size=bs, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=True)
    dann = DANN(model, feat_dim=512).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(dann.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    tgt_iter = cycle(tgt_train_loader)
    total_steps, step = epochs * len(source_loader), 0
    for epoch in range(epochs):
        dann.train()
        for images_s, labels_s in tqdm(source_loader, desc=f"[{setting}] DANN ep{epoch+1}/{epochs}"):
            images_s, labels_s = images_s.to(device), labels_s.to(device)
            images_t, _ = next(tgt_iter)
            images_t = images_t.to(device)
            lambd = 2.0 / (1.0 + math.exp(-10.0 * step / max(total_steps, 1))) - 1.0
            optimizer.zero_grad()
            out_s, dom_s = dann(images_s, lambd)
            out_t, dom_t = dann(images_t, lambd)
            d_s = torch.zeros(images_s.size(0), dtype=torch.long, device=device)
            d_t = torch.ones(images_t.size(0), dtype=torch.long, device=device)
            loss = (criterion(out_s, labels_s)
                    + criterion(dom_s, d_s) + criterion(dom_t, d_t)
                    + lam_im * information_maximization_loss(out_t))
            loss.backward()
            optimizer.step()
            step += 1
        scheduler.step()
    adapt_bn(model, tgt_eval_loader)   # `model` shares weights with `dann`
    return model


# 2c. Class-balanced self-training
#   Pick the top-`frac` most confident target samples PER predicted class (balanced,
#   works even when MixUp/label-smoothing keep absolute confidence low), then fine-tune
#   the model on them jointly with the labeled source. Iterated over a few rounds.
@torch.no_grad()
def balanced_select(probs, frac):
    conf, pred = probs.max(dim=1)
    idx_sel, lab_sel = [], []
    for c in range(NUM_CLASSES):
        ids = (pred == c).nonzero(as_tuple=False).squeeze(1)
        if ids.numel() == 0:
            continue
        k = max(1, int(frac * ids.numel()))
        top = ids[conf[ids].argsort(descending=True)[:k]]
        idx_sel += top.tolist()
        lab_sel += [c] * top.numel()
    return idx_sel, lab_sel


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    return torch.cat([torch.softmax(model(im.to(device)), dim=1).cpu() for im, _ in loader])


def selftrain(model, setting, rounds=3, frac=0.5, ft_epochs=10, lr=0.01):
    cfg = CONFIGS[setting]
    bs = cfg["batch"]
    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])
    _, tgt_aug, tgt_eval_loader = target_loaders(setting, bs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for r in range(rounds):
        idx, plabels = balanced_select(predict_probs(model, tgt_eval_loader), frac)
        pseudo_loader = DataLoader(PseudoLabeledSubset(tgt_aug, idx, plabels), batch_size=bs,
                                   shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ft_epochs)
        pseudo_iter = cycle(pseudo_loader)
        for epoch in range(ft_epochs):
            model.train()
            for images_s, labels_s in tqdm(source_loader,
                                           desc=f"[{setting}] ST r{r+1} ft{epoch+1}/{ft_epochs}"):
                images_s, labels_s = images_s.to(device), labels_s.to(device)
                images_t, labels_t = next(pseudo_iter)
                images_t, labels_t = images_t.to(device), labels_t.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images_s), labels_s) + criterion(model(images_t), labels_t)
                loss.backward()
                optimizer.step()
            scheduler.step()
    adapt_bn(model, tgt_eval_loader)
    return model


# 3. Ensemble distillation
#   Average target predictions from a diverse set of from-scratch teachers into clean,
#   class-balanced pseudo-labels; train the ResNet-34 student on them. Because the
#   averaged labels are cleaner than any single teacher, the student exceeds them all.
@torch.no_grad()
def ensemble_probs(teacher_paths, loader):
    probs_sum = None
    for path, arch in teacher_paths:
        m = build_backbone(arch).to(device)
        m.load_state_dict(torch.load(path))
        m.eval()
        ps = torch.cat([torch.softmax(m(im.to(device)), dim=1).cpu() for im, _ in loader])
        probs_sum = ps if probs_sum is None else probs_sum + ps
        del m
        torch.cuda.empty_cache()
    return probs_sum / len(teacher_paths)


def ensemble_distill(setting, teacher_paths, student_init, rounds=2, frac=0.5,
                     ft_epochs=10, lr=0.01):
    cfg = CONFIGS[setting]
    bs = cfg["batch"]
    source_loader, val_loader, _ = build_loaders(setting, bs, cfg["balanced"])
    _, tgt_aug, tgt_eval_loader = target_loaders(setting, bs)

    avg = ensemble_probs(teacher_paths, tgt_eval_loader)     # fixed ensemble -> distillation
    idx, plabels = balanced_select(avg, frac)
    print(f"[{setting}] ensemble pseudo: {len(idx)}/{len(tgt_aug)} "
          f"({100*len(idx)/len(tgt_aug):.1f}%), classes {len(set(plabels))}/{NUM_CLASSES}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(student_init))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for r in range(rounds):
        pseudo_loader = DataLoader(PseudoLabeledSubset(tgt_aug, idx, plabels), batch_size=bs,
                                   shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ft_epochs)
        pseudo_iter = cycle(pseudo_loader)
        for epoch in range(ft_epochs):
            model.train()
            for images_s, labels_s in tqdm(source_loader,
                                           desc=f"[{setting}] distill r{r+1} ft{epoch+1}/{ft_epochs}"):
                images_s, labels_s = images_s.to(device), labels_s.to(device)
                images_t, labels_t = next(pseudo_iter)
                images_t, labels_t = images_t.to(device), labels_t.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images_s), labels_s) + criterion(model(images_t), labels_t)
                loss.backward()
                optimizer.step()
            scheduler.step()
    adapt_bn(model, tgt_eval_loader)
    return model


# Full pipeline for one setting
#   (a) train a diverse pool of refined teachers (varied seeds; a ResNet-50 for C->P),
#   (b) ensemble-distill them into the final ResNet-34 student, iterating with the
#       student added back as a teacher.
# More seeds -> more diversity -> higher accuracy (diminishing). Tune TEACHER_SEEDS /
# DISTILL_ITERS for the compute budget.
TEACHER_SEEDS = [42, 1, 7, 13, 21]
DISTILL_ITERS = 3


def make_refined_teacher(setting, seed, arch="resnet34"):
    model = train_source(setting, seed=seed, arch=arch)
    model = entropy_refine(model, setting)
    model = dann_refine(model, setting)
    model = selftrain(model, setting)
    path = os.path.join(CKPT_DIR, f"{setting}_teacher_s{seed}_{arch}.pth")
    torch.save(model.state_dict(), path)
    print(f"[{setting}] teacher seed{seed}/{arch}: target acc {eval_target(model, setting):.2f}% "
          f"-> {path}")
    return (path, arch)


def run_setting(setting):
    teachers = [make_refined_teacher(setting, s, "resnet34") for s in TEACHER_SEEDS]
    if setting == "CtoP":   # add a from-scratch ResNet-50 teacher (teacher only)
        teachers.append(make_refined_teacher(setting, 42, "resnet50"))

    # Strongest teacher (by source-val proxy / first) seeds the student; iterate distill.
    student_path = teachers[0][0]
    for it in range(DISTILL_ITERS):
        student = ensemble_distill(setting, teachers, student_path)
        student_path = os.path.join(CKPT_DIR, f"{setting}_student_it{it+1}.pth")
        torch.save(student.state_dict(), student_path)
        teachers.append((student_path, "resnet34"))   # student becomes a teacher
        print(f"[{setting}] distill iter {it+1}: target acc {eval_target(student, setting):.2f}%")
    torch.save(torch.load(student_path), CKPT_PATHS[setting])


# Train both settings and save the final ResNet-34 checkpoints.
if __name__ == "__main__":
    run_setting("CtoP")
    run_setting("PtoC")

    # =========================================================================
    # 5. Evaluation and Submit
    #   [Setting1] CtoP : Evaluate on CUB-200-Paintings.
    #   [Setting2] PtoC : Evaluate on CUB-200.
    # Logic kept identical to the starter "DO NOT CHANGE" cells. `model` is a ResNet-34
    # put in eval mode so the target-adapted BatchNorm running stats are used.
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
