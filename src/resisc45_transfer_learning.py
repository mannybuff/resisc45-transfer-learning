"""
resisc45_transfer_learning.py

A clean, portfolio-style example of transfer learning on the
NWPU-RESISC45 remote sensing scene classification dataset.

This script focuses on a single, clear baseline:

- Pretrained ResNet-50 backbone (ImageNet weights)
- New classification head for 45 scene classes
- Standard train/validation/test loop
- Accuracy and confusion-matrix reporting

It is meant to be readable and easy to adapt, not hyper-optimized.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report


@dataclass
class TrainConfig:
    data_root: Path
    train_dir_name: str = "train"
    val_dir_name: str = "val"
    test_dir_name: str = "test"
    num_classes: int = 45
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    freeze_backbone: bool = True


def set_seed(seed: int) -> None:
    import random
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Expects the dataset to be organized as:

        data_root/
            train/
                class_1/
                class_2/
                ...
            val/
                class_1/
                ...
            test/
                class_1/
                ...

    Each class_* folder contains its images.
    """
    input_size = 224

    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dir = cfg.data_root / cfg.train_dir_name
    val_dir = cfg.data_root / cfg.val_dir_name
    test_dir = cfg.data_root / cfg.test_dir_name

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transforms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    print(f"Number of classes inferred from train set: {len(train_ds.classes)}")

    return train_loader, val_loader, test_loader


def build_model(cfg: TrainConfig) -> nn.Module:
    """
    Build a ResNet-50 model with a new classification head for NWPU-RESISC45.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, cfg.num_classes),
    )

    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def train_model(cfg: TrainConfig) -> Tuple[nn.Module, dict]:
    set_seed(cfg.seed)

    device = cfg.device
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{cfg.num_epochs} "
            f"| train_loss={train_loss:.4f}, train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Track best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = cfg.data_root / "resisc45_resnet50_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved to {best_path}")

    # Load best weights (if any)
    if (cfg.data_root / "resisc45_resnet50_best.pth").exists():
        model.load_state_dict(torch.load(cfg.data_root / "resisc45_resnet50_best.pth", map_location=device))

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"
Final test performance: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # Detailed classification report
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    print("
Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Optionally, save confusion matrix as a .npy for later plotting
    cm_path = cfg.data_root / "resisc45_resnet50_confusion.npy"
    np.save(cm_path, cm)
    print(f"Confusion matrix saved to {cm_path}")

    return model, history


if __name__ == "__main__":
    # Example usage:
    # Point data_root to the folder that contains train/val/test subfolders.
    cfg = TrainConfig(
        data_root=Path("/path/to/NWPU-RESISC45-split"),
        num_classes=45,
        batch_size=32,
        num_epochs=20,
        freeze_backbone=True,
    )
    train_model(cfg)
