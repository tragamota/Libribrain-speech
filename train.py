import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from torch import autocast, GradScaler
from torch.utils.data import DataLoader

from pnpl.datasets import LibriBrainSpeech

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import SupConLoss

from torchmetrics import Accuracy, F1Score, AUROC
from tqdm import tqdm

from augmentations import MEGCompose, GaussianNoise, UniformScaling, ChannelDropout
from speechclassifier import SpeechClassifier

SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
                       146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]


def normalize_per_sensor(x, eps=1e-6):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + eps

    return (x - mean) / std

def train(model, dataloader, optimizer, criterion, scaler, device):
    model.train()

    supcon_loss_fn = SupConLoss(temperature=0.2, distance=CosineSimilarity()).to(device)

    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    auc_metric = AUROC(task="multiclass", num_classes=2).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=2).to(device)

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    transforms = MEGCompose([
        GaussianNoise(std=0.05, p=0.5),
        UniformScaling(scale_min=0.8, scale_max=1.2, p=0.5),
        ChannelDropout(dropout_prob=0.3, p=0.35),
    ]).to(device)

    for data, targets in progress_bar:
        # data = data[:, SENSORS_SPEECH_MASK, :]
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        data = transforms(data)
        # data = normalize_per_sensor(data)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(data)

            if model.mode == "contrastive":
                preds, embeddings = out
                preds = preds.permute(0, 2, 1)

                loss_ce = criterion(preds, targets)

                embeddings = F.normalize(embeddings, p=2, dim=-1).view(-1, embeddings.shape[-1])
                loss_supcon = supcon_loss_fn(embeddings, targets.view(-1))
                loss = loss_ce + loss_supcon
            else:
                preds = out
                preds = preds.permute(0, 2, 1)

                loss_ce = criterion(preds, targets)

                loss = loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = preds.detach()

        accuracy_metric.update(preds, targets)
        auc_metric.update(preds, targets)
        f1_metric.update(preds, targets)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy_metric.compute().item():.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_acc = accuracy_metric.compute().item()
    avg_auc = auc_metric.compute().item()
    avg_f1 = f1_metric.compute().item()

    accuracy_metric.reset()
    auc_metric.reset()
    f1_metric.reset()

    return avg_loss, avg_acc, avg_auc, avg_f1

def eval(model, dataloader, criterion, device):
    model.eval()

    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    auc_metric = AUROC(task="multiclass", num_classes=2).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=2).to(device)

    total_loss = 0.0

    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Validating", leave=True):
            # data = data[:, SENSORS_SPEECH_MASK, :]

            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if model.mode == "contrastive":
                preds, embeddings = model(data)
            else:
                preds = model(data)

            preds = preds.permute(0, 2, 1)

            loss = criterion(preds, targets)

            total_loss += loss.item()

            accuracy_metric.update(preds, targets)
            auc_metric.update(preds, targets)
            f1_metric.update(preds, targets)

    avg_loss = total_loss / len(dataloader)
    avg_acc = accuracy_metric.compute().item()
    avg_auc = auc_metric.compute().item()
    avg_f1 = f1_metric.compute().item()

    accuracy_metric.reset()
    auc_metric.reset()
    f1_metric.reset()

    return avg_loss, avg_acc, avg_auc, avg_f1

def main(args):
    os.makedirs(args.output, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = SpeechClassifier(mode="classification").to(args.device)

    train_dataset = LibriBrainSpeech("./datasets", partition="train", tmin=0.0, tmax=0.8, oversample_silence_jitter=50)
    val_dataset = LibriBrainSpeech("./datasets", partition="validation", tmin=0.0, tmax=0.8, stride=5)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, persistent_workers=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=2, persistent_workers=True, shuffle=False)

    weight = torch.tensor([2.23, 0.64])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss(weight=None).to(args.device)

    scaler = GradScaler(enabled=True)

    best_val_loss = np.inf

    for epoch in range(args.epochs):
        train_loss, train_acc, train_auc, train_f1 = train(model, train_dataloader, optimizer, criterion, scaler, args.device)
        val_loss, val_acc, val_auc, val_f1 = eval(model, val_dataloader, criterion, args.device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f" Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")
        print(f" Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output, "best.pt"))
            print(f"✅ Saved new best model weights at epoch {epoch + 1} with val loss {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.output, "last.pt"))


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("--lr", type=int, default=3e-5)
    argparser.add_argument("--epochs", type=int, default=20)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--num_workers", type=int, default=2)
    argparser.add_argument("--seed", type=int, default=374)
    argparser.add_argument("--output", type=str, default="weights")
    argparser.add_argument("--device", type=str, default="cuda")

    main(argparser.parse_args())
