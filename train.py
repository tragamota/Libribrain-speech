import torch
from torchmetrics import Accuracy

from tqdm import tqdm
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from pnpl.datasets import LibriBrainSpeech

from augmentations import MEGCompose, GaussianNoise, UniformScaling, ChannelDropout
from speechclassifier import SpeechClassifier


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    transforms = MEGCompose([
        GaussianNoise(std=0.01, p=0.7),
        UniformScaling(scale_min=0.9, scale_max=1.1, p=0.35),
        ChannelDropout(dropout_prob=0.1, p=0.4),
    ]).to(device)

    for data, labels in progress_bar:
        data = data.to(device)
        labels = labels.to(device)

        data = transforms(data)

        preds = model(data)
        loss = criterion(preds.permute(0, 2, 1), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pred_classes = preds.argmax(dim=-1)
        accuracy_metric.update(pred_classes, labels)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy_metric.compute().item():.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_acc = accuracy_metric.compute().item()

    return avg_loss, avg_acc

def eval(model, dataloader, criterion, device):
    model.eval()

    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    total_loss = 0.0

    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Validating", leave=True):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(data)
            loss = criterion(preds.permute(0, 2, 1), labels)

            total_loss += loss.item()
            pred_classes = preds.argmax(dim=-1)
            accuracy_metric.update(pred_classes, labels)

    avg_loss = total_loss / len(dataloader)
    avg_acc = accuracy_metric.compute().item()

    return avg_loss, avg_acc

def main(args):
    model = SpeechClassifier(mode="classification").to(args.device)

    transforms = MEGCompose([
        GaussianNoise(std=0.01, p=0.7),
        UniformScaling(scale_min=0.8, scale_max=1.25, p=0.35),
        ChannelDropout(dropout_prob=0.1, p=0.4),
    ])

    train_dataset = LibriBrainSpeech("./datasets", partition="train", tmin=0.0, tmax=1.25, oversample_silence_jitter=15, stride=150)
    val_dataset = LibriBrainSpeech("./datasets", partition="validation", tmin=0.0, tmax=1.0, stride=5)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                  pin_memory_device=args.device, persistent_workers=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True,
                                pin_memory_device=args.device, persistent_workers=True, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2).to(args.device)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, args.device)
        val_loss, val_acc = eval(model, val_dataloader, criterion, args.device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"✅Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"🔍Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("--lr", type=int, default=1e-4)
    argparser.add_argument("--epochs", type=int, default=20)
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--num_workers", type=int, default=6)
    argparser.add_argument("--device", type=str, default="cuda")

    main(argparser.parse_args())
