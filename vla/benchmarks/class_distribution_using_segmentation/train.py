import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from config import TrainConfig, ConfigPaths, construct_configs
from manager import DatasetManager, make_dataloaders
from model import SimpleClassifier
from features import FeatureStore
from precompute import precompute_features

ttype = torch.float32

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
    save_dir="./checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=TrainConfig.learning_rate,
        momentum=TrainConfig.momentum
    )

    criterion = nn.KLDivLoss(reduction='batchmean')

    best_loss = float("inf")
    best_path = os.path.join(save_dir, "best_classifier.pt")
    last_path = os.path.join(save_dir, "last_classifier.pt")

    for epoch in range(TrainConfig.epochs):
        classifier.train()
        total_loss = 0
        torch.cuda.empty_cache()
        for labels, images in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(device), labels.to(ttype).to(device)
            optimizer.zero_grad()

            if train_loader.dataset.feature_store is None:
                feats = model.encode(images)
            else:
                feats = images
            feats = feats.to(ttype)
            out = classifier(feats)

            log_probs = F.log_softmax(out, dim=1)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        avg_train = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train}")

        if epoch % 5 == 0:
            print(f"Testing, epoch {epoch+1}")
            classifier.eval()
            total_test = 0
            with torch.no_grad():
                for labels, images in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    if test_loader.dataset.feature_store is None:
                        feats = model.encode(images)
                    else:
                        feats = images
                    feats = feats.to(ttype)
                    out = classifier(feats)
                    log_probs = F.log_softmax(out, dim=1)
                    total_test += criterion(log_probs, labels).item()

            avg_test = total_test / len(test_loader)
            print(f"[Epoch {epoch+1}] Test  Loss: {avg_test}")

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

    print(f"Training finished. Best test loss = {best_loss}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")

    return classifier

def score_loss(model, classifier, loader, device):
    criterion = nn.KLDivLoss(reduction='batchmean')
    with torch.no_grad():
        sum_loss = 0
        for y, img in tqdm(loader):
            if loader.dataset.feature_store is None:
                feats = model.encode(img.to(device))
            else:
                feats = img.to(device)
            feats = feats.to(ttype)
            out = classifier(feats)
            log_probs = F.log_softmax(out, dim=1)
            loss = criterion(log_probs, y.to(device))
            sum_loss += loss.item()
        sum_loss /= len(loader)

    print("Average loss:", sum_loss)

def symmetric_topk_mass_recall(logits, targets, k=10):

    probs = F.softmax(logits, dim=-1)

    topk_pred_idx = probs.topk(k, dim=-1).indices
    target_mass_in_pred_topk = torch.gather(targets, dim=-1, index=topk_pred_idx)
    recall_p_to_q = target_mass_in_pred_topk.sum(dim=-1)

    topk_target_idx = targets.topk(k, dim=-1).indices
    pred_mass_in_target_topk = torch.gather(probs, dim=-1, index=topk_target_idx)
    recall_q_to_p = pred_mass_in_target_topk.sum(dim=-1)

    symmetric_recall = 0.5 * (recall_p_to_q + recall_q_to_p)

    return symmetric_recall.mean()

def symmetric_topk_recall_score(model, classifier, loader, device):
    with torch.no_grad():
        average_recall = 0
        for y, img in tqdm(loader):
            if loader.dataset.feature_store is None:
                feats = model.encode(img.to(device))
            else:
                feats = img.to(device)
            feats = feats.to(ttype)
            out = classifier(feats).squeeze()
            y = y.squeeze().to(device)
            recall = symmetric_topk_mass_recall(out, y, k=17)
            average_recall += recall.item()
        average_recall /= len(loader)
    print("Average symmetric top-k recall:", average_recall)

def distribution_iou(logits, targets):
    probs = F.softmax(logits, dim=-1)

    intersection = torch.minimum(probs, targets).sum(dim=-1)
    union = torch.maximum(probs, targets).sum(dim=-1)

    return (intersection / (union + 1e-8)).mean()

def IoU_score(model, classifier, loader, device):
    with torch.no_grad():
        average_recall = 0
        for y, img in tqdm(loader):
            if loader.dataset.feature_store is None:
                feats = model.encode(img.to(device))
            else:
                feats = img.to(device)
            feats = feats.to(ttype)
            out = classifier(feats).squeeze()
            y = y.squeeze().to(device)
            recall = distribution_iou(out, y)
            average_recall += recall.item()
        average_recall /= len(loader)
    print("Average IoU score:", average_recall)

def benchmark(model, preprocessor, train_json="train_los_dataset.json", test_json="test_los_dataset.json",
              use_precomputed_features=True, random_seed=None,
              generalization_set_folder="", config_path="example_config.json"):

    with open(config_path, 'r') as file:
        config_dict = json.load(file)
        construct_configs(**config_dict)
    print(f"{TrainConfig.learning_rate=}; {TrainConfig.momentum=}; {TrainConfig.epochs=}; "
          f"{TrainConfig.output_activation=}; {TrainConfig.num_layers=}; {TrainConfig.layers_sizes=}\n")
    dataset_manager = None
    if random_seed is not None:
        dataset_manager = DatasetManager(ConfigPaths.path_to_raw_data, random_seed=random_seed)

    generalization_dataset_manager = None
    if generalization_set_folder != "":
        generalization_dataset_manager = DatasetManager(generalization_set_folder, random_seed=random_seed,
                                                        full_folder=True)

    store = FeatureStore(os.path.join(ConfigPaths.feature_store_path, type(model).__name__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, score_loader, generalization_loader, num_classes = \
        make_dataloaders(
            train_json,
            test_json,
            preprocessor,
            feature_store=(store if use_precomputed_features else None),
            workers=8,
            dataset_manager=dataset_manager,
            generalization_dataset_manager=generalization_dataset_manager,
            batch=32
        )

    if use_precomputed_features:
        print("[Benchmark] Precomputing features...")
        precompute_features(
            model=model,
            dataset=train_loader.dataset,
            store=store,
            device=device
        )
        precompute_features(
            model=model,
            dataset=test_loader.dataset,
            store=store,
            device=device
        )
        if generalization_dataset_manager is not None:
            precompute_features(
                model=model,
                dataset=generalization_loader.dataset,
                store=store,
                device=device
            )

    if use_precomputed_features:
        feat_dim = train_loader.dataset[0][1].flatten().shape[0]
    else:
        feat_dim = extract_features_size(model, train_loader, device)

    classifier = SimpleClassifier(feat_dim, num_classes, TrainConfig.output_activation,
                                  num_hidden_layers=TrainConfig.num_layers,
                                  hidden_size=TrainConfig.layers_sizes).to(device).to(ttype)

    classifier = train_classifier(model, classifier, train_loader, test_loader, device)
    print("Computing score on test set:\n")
    symmetric_topk_recall_score(model, classifier, score_loader, device)
    IoU_score(model, classifier, score_loader, device)
    score_loss(model, classifier, score_loader, device)
    if generalization_dataset_manager is not None:
        print("Computing score on generalization set:\n")
        symmetric_topk_recall_score(model, classifier, generalization_loader, device)
        IoU_score(model, classifier, generalization_loader, device)
        score_loss(model, classifier, generalization_loader, device)
