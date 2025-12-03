import os
import torch
from torch import nn
from tqdm import tqdm
from config import TrainConfig, LabelType
from manager import DatasetManager
from model import SimpleClassifier


def extract_feature_dim(model, loader, device):
    for _, (_, img) in enumerate(loader):
        with torch.no_grad():
            feat = model.encode(img.to(device))
        return feat.flatten().shape[0]


def extract_features_size(model, train_loader, device):
    for _, data in enumerate(train_loader, 0):
        _, img = data
        img = img[0, :, :, :].unsqueeze(0).to(device)
        features = model.encode(img)
        return features.flatten().size()[0]


def train_classifier(
    model,
    classifier,
    train_loader,
    test_loader,
    device,
    label_type: LabelType,
    save_dir="./checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=TrainConfig.learning_rate,
        momentum=TrainConfig.momentum
    )

    if label_type == LabelType.DISTANCE:
        criterion = nn.MSELoss()
    elif label_type == LabelType.TYPE_CLASSIFICATION:
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported label type: {label_type}")

    best_loss = float("inf")
    best_path = os.path.join(save_dir, "best_classifier.pt")
    last_path = os.path.join(save_dir, "last_classifier.pt")

    for epoch in range(TrainConfig.epochs):
        classifier.train()
        total_loss = 0

        for labels, images in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            feats = model.encode(images)
            out = classifier(feats)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train:.4f}")

        classifier.eval()
        total_test = 0
        with torch.no_grad():
            for labels, images in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = classifier(model.encode(images))
                total_test += criterion(out, labels).item()

        avg_test = total_test / len(test_loader)
        print(f"[Epoch {epoch+1}] Test  Loss: {avg_test:.4f}")

        if avg_test < best_loss:
            best_loss = avg_test
            torch.save({
                "epoch": epoch,
                "model_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "test_loss": best_loss
            }, best_path)
            print(f"Saved BEST checkpoint at epoch {epoch+1} → {best_path}")

        # Save last epoch (always)
        torch.save({
            "epoch": epoch,
            "model_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "test_loss": avg_test
        }, last_path)

    print(f"Training finished. Best test loss = {best_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")

    return classifier


def score(model, classifier, loader, label_type: LabelType, device):
    if label_type == LabelType.TYPE_CLASSIFICATION:
        correct = 0
        with torch.no_grad():
            for y, img in loader:
                out = classifier(model.encode(img.to(device)))
                if out.argmax() == y.argmax():
                    correct += 1
        print("Accuracy:", correct / len(loader))

    elif label_type == LabelType.DISTANCE:
        mse = 0
        loss = nn.MSELoss()
        with torch.no_grad():
            for y, img in loader:
                out = classifier(model.encode(img.to(device)))
                mse += loss(out, y.to(device)).item()
        print("Average MSE:", mse / len(loader))

    else:
        raise ValueError(f"Unsupported label type {label_type}")


def benchmark(model, preprocessor, train_json, test_json, label_type: LabelType):
    manager = DatasetManager(
        directory="../../../Mountain_Range",
        label_type=label_type
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, score_loader, num_classes = \
        manager.make_dataloaders(train_json, test_json, preprocessor)

    feat_dim = extract_features_size(model, train_loader, device)
    classifier = SimpleClassifier(feat_dim, num_classes, TrainConfig.output_activation).to(device)

    classifier = train_classifier(model, classifier, train_loader, test_loader, device, label_type)
    score(model, classifier, score_loader, label_type, device)

