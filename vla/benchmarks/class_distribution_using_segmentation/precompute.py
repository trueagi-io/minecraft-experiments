import torch
from tqdm import tqdm

def precompute_features(model, dataset, store, device):
    """
    Run the model on all images once and cache features.
    dataset.items = list of (label, path_stem)
    """

    # model = model.to(device)
    # model.eval()

    print("\n[Precompute] Starting feature extraction…\n")

    with torch.no_grad():
        for img_path, _ in tqdm(dataset.items):
            feature_path = str(img_path).strip(".png").replace("/", "_")
            if store.exists(feature_path):
                continue

            # load image
            img = dataset.load_image(img_path).to(device)

            # run feature extractor
            features = model.encode(img.unsqueeze(0)).squeeze()

            # save features
            store.save(feature_path, features)

    print("\n[Precompute] Feature extraction complete.\n")
