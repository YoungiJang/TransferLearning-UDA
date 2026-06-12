"""
DL2 Homework 3: Transfer Learning - Unsupervised Domain Adaptation (UDA)

Local (non-Colab) version converted from DL_HW3.ipynb.

- Unsupervised Domain Adaptation assumes access to LABELED source and UNLABELED target.
    * During training: labeled source images + unlabeled target images.
    * During evaluation: tested on target images, compared with ground-truth target labels.

Settings:
    [Setting1] CtoP : Source = CUB-200,          Target = CUB-200-Paintings
    [Setting2] PtoC : Source = CUB-200-Paintings, Target = CUB-200

IMPORTANT constraints (from the assignment):
    * No pretrained weights.
    * No target labels during training (no early stopping / tuning / model selection on target).
    * The SAME architecture must be used for both settings.
    * The [Evaluation and Submit] section must NOT be modified (kept identical below).
"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# Paths / environment  (was: Colab drive.mount + /content/drive/MyDrive/DL_HW3)
# Edited for local execution. The datasets are already downloaded & extracted
# into this directory, so no wget/gdown/drive.mount is needed.
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = BASE_DIR
os.makedirs(output_dir, exist_ok=True)

# Source folders (extracted locally).
CUB_200_PATH = os.path.join(BASE_DIR, "CUB_200_2011", "images")
# NOTE: the Paintings zip extracts to "CUB-200-Painting" (hyphens, singular).
CUB_200_PAINTINGS_PATH = os.path.join(BASE_DIR, "CUB-200-Painting")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. Model Architecture
# (Starter CNN. You may customize it, but you must NOT use pretrained weights
#  and you MUST keep the same architecture across both settings.)
# =============================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 24 * 24, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 200)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# Unlabeled target dataset wrapper (drops labels)
# =============================================================================
class UnlabeledDataset(Dataset):
    def __init__(self, labeled_dataset):
        self.dataset = labeled_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image


# =============================================================================
# 3. Training configuration & data loaders
# You can set different configs per setting (they need not be identical).
# =============================================================================
def build_loaders(setting, batch_size):
    cub_200_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # === YOU CAN ADD SOME AUGMENTATIONS === #
    ])

    cub_200_paintings_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.7815, 0.7699, 0.7322), (0.2654, 0.2694, 0.2941)),
        # === YOU CAN ADD SOME AUGMENTATIONS === #
    ])

    cub_dataset = ImageFolder(root=CUB_200_PATH, transform=cub_200_transform)
    cub_paintings_dataset = ImageFolder(root=CUB_200_PAINTINGS_PATH,
                                        transform=cub_200_paintings_transform)

    if setting == "CtoP":      # [Setting1]
        source_dataset = cub_dataset
        target_dataset = UnlabeledDataset(cub_paintings_dataset)  # labels absent
    elif setting == "PtoC":    # [Setting2]
        source_dataset = cub_paintings_dataset
        target_dataset = UnlabeledDataset(cub_dataset)            # labels absent
    else:
        raise ValueError(setting)

    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=4)
    return source_loader, target_loader


# =============================================================================
# 4. Train the Network
# Same architecture for both settings. You may choose optimizer, objective,
# augmentations, replay buffer, joint vs sequential training, etc.
#
# [IMPORTANT] Target labels must NOT be used during training (no early
# stopping / tuning / model selection on target accuracy).
# =============================================================================
def train(setting, num_epochs, learning_rate, batch_size):
    model = Net().to(device)
    source_loader, target_loader = build_loaders(setting, batch_size)

    # --- [Train 1/2] Supervised training on labeled source ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        n_iter, loss_total, acc_total = 0, 0.0, 0.0
        for data in tqdm(source_loader, desc=f"[{setting}] src ep{epoch + 1}"):
            inputs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=-1)
            acc = pred.eq(targets).float().mean()
            loss_total += loss.item()
            acc_total += acc.item()
            n_iter += 1
        print(f"Epoch: {epoch + 1} - Loss: {loss_total / n_iter:.3f} "
              f"- Acc: {acc_total / n_iter:.3f}")
    print("[Train 1/2] Finished Training on Source Data.")

    # --- [Train 2/2] Unsupervised training on unlabeled target (InfoNCE) ---
    from info_nce import info_nce  # pip install info-nce-pytorch

    contrastive_temperature = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        n_iter, loss_total = 0, 0.0
        for inputs in tqdm(target_loader, desc=f"[{setting}] tgt ep{epoch + 1}"):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            inputs_view1 = inputs
            inputs_view2 = inputs + torch.randn_like(inputs) * 0.05

            query_features = model(inputs_view1)
            positive_key_features = model(inputs_view2)

            loss = info_nce(query_features, positive_key_features,
                            temperature=contrastive_temperature)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            n_iter += 1
        print(f"Epoch: {epoch + 1} - InfoNCE Loss: {loss_total / n_iter:.3f}")
    print("[Train 2/2] Finished Training on Target Data")

    return model


# =============================================================================
# Run training for both settings and save checkpoints.
# Configs can differ per setting.
# =============================================================================
CtoP_CKPT_PATH = os.path.join(output_dir, "CtoP.pth")
PtoC_CKPT_PATH = os.path.join(output_dir, "PtoC.pth")
BATCH_SIZE = 64

if __name__ == "__main__":
    # ---- [Setting1] CtoP ----
    model = train(setting="CtoP", num_epochs=7, learning_rate=0.001,
                  batch_size=BATCH_SIZE)
    torch.save(model.state_dict(), CtoP_CKPT_PATH)

    # ---- [Setting2] PtoC ----
    model = train(setting="PtoC", num_epochs=7, learning_rate=0.001,
                  batch_size=BATCH_SIZE)
    torch.save(model.state_dict(), PtoC_CKPT_PATH)

    # =========================================================================
    # 5. Evaluation and Submit
    #   [Setting1] CtoP : Evaluate on CUB-200-Paintings.
    #   [Setting2] PtoC : Evaluate on CUB-200.
    # The logic below is kept identical to the starter notebook's
    # "DO NOT CHANGE" cells (only Colab paths via variables differ).
    # =========================================================================
    model = Net().to(device)

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
    model.eval()
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
    model.eval()
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
