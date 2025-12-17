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
        for label, path in tqdm(dataset.items):
            if store.exists(path):
                continue

            # load image
            img = dataset.load_image(path).to(device)

            # run feature extractor
            features = model.encode(img)

            # save features
            store.save(path, features)

    print("\n[Precompute] Feature extraction complete.\n")
