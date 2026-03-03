import torch
from tqdm import tqdm
import torch.nn.functional as F

def infer_model(classifier, device, img, loader, model, ttype):
    if loader.dataset.feature_store is None:
        feats = model.encode(img.to(device))
    else:
        feats = img.to(device)
    feats = feats.to(ttype)
    out = classifier(feats)
    return out

def score_loss(model, classifier, loader, device, ttype):
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    with torch.no_grad():
        sum_loss = 0
        for y, img in tqdm(loader):
            out = infer_model(classifier, device, img, loader, model, ttype)
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

def symmetric_topk_recall_score(model, classifier, loader, device, ttype):
    with torch.no_grad():
        average_recall = 0
        for y, img in tqdm(loader):
            out = infer_model(classifier, device, img, loader, model, ttype).squeeze()
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

def IoU_score(model, classifier, loader, device, ttype):
    with torch.no_grad():
        average_recall = 0
        for y, img in tqdm(loader):
            out = infer_model(classifier, device, img, loader, model, ttype).squeeze()
            y = y.squeeze().to(device)
            recall = distribution_iou(out, y)
            average_recall += recall.item()
        average_recall /= len(loader)
    print("Average IoU score:", average_recall)