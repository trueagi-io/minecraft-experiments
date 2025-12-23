#  Visual feature benchmarking tools for Minecraft

##  Summary Table

| Module      | Description |
|-------------|-------------|
| **train.py** | Runs training, selects criterion, evaluates, saves checkpoints |
| **model.py** | Classifier MLP |
| **manager.py** | Orchestrates dataset creation and dataloaders |
| **builder.py** | Builds datasets from raw folder structure |
| **dataset.py** | Defines PyTorch Dataset class |
| **config.py** | Global configuration and constants |
| **feaures.py** | Storage for precomputed features |
| **precompute.py** | Feature precompute procedure |

---

## 🟦 **train.py**
### **Purpose:**  
Implements the **training and evaluation pipeline**.

### **Responsibilities:**
- Loads training and testing dataloaders via `DatasetManager`.
- Chooses the correct **loss function** based on label type (regression/classification).
- Trains a classifier on top of frozen feature-extractor embeddings.
- Tracks and saves:
  - `best_classifier.pth` (best validation loss)
  - `current_classifier.pth` (latest snapshot)
- Computes final performance:
  - MSE for distance regression
  - Accuracy for type classification
 - Provides helper function:
   - `extract_features_size(model, loader)` – determines the classifier input dimension dynamically.
- For benchmark purposes contains `benchmark` function
  - Required input parameters:
    - **model** is the model which used to extract features from the image tensor. Must contain `encode` function which takes image tensor as input and outputs features.
    - **preprocessor** is the preprocessor sequence which is applied to the input image. Preprocessor sequence must contain `T.ToTensor()` and may contain resize, normalize etc.
    - either **train_json** and **test_json** or **random_seed**. In one case, dataset is pre-prepared and saved into json files (see **dataset_index_creation_example.py**). In case of **random_seed** as input dataset is being created in the runtime so no need in json files.
  - Optional input parameters:
    - **label_type** either `LabelType.DISTANCE` or `LabelType.TYPE_CLASSIFICATION` (from `config.py`). Depends on this parameter, classifier or regression model will be created and trained.
    - **use_precomputed_features** if set to True, features for all input images will be pre-computed and saved to prevent feature computing while train and test. That helps to reduce time needed to run benchmark. Though requires lots of disk space (depends on which features are used).
    - **generalization_set_folder** is used to test generalization capability of the trained model. This folder will be used only to compute accuracy/mse in the end of training process.
    - **config_path** path to the json file which contains parameters used to create and train model. Also contains pathes to the dataset folder and folder which will contain precomputed features. See `example.json`.
---

## 🟩 **model.py**
### **Purpose:**  
Defines **all models used in the system**, including:

### **Responsibilities:**
- The **SimpleClassifier** (MLP):
  - Supports configurable number of layers
  - Supports multiple output activations

---

## 🟧 **manager.py**
### **Purpose:**  
Central module that orchestrates dataset creation and loading.

### **Responsibilities:**
- Exposes `DatasetManager`, a high-level interface for:
  - Building a dataset from raw image folders.
  - Reading generated JSON dataset files.
  - Creating PyTorch dataloaders (train/test/score).
- Handles both **distance regression** and **type classification** dataset types.
- Utility:
  - `flatten_class_map()` converts `{class_name: [paths...]}` into index-based dataset lists.
- Backwards-compatibility wrapper:
  - `CreateDataloader()` matches our older function signature.

---

## 🟨 **builder.py**
### **Purpose:**  
Creates training/test datasets **from raw images and metadata**.

### **Responsibilities:**
- Contains the abstract base:  
  **`DatasetBuilder`**
  - Extracts image paths + line-of-sight metadata.
  - Filters out invalid or missing data.
- Two concrete builders:

### `DistanceDatasetBuilder`
- Builds **balanced regression dataset**.
- Performs histogram binning of distances.
- Samples uniformly from usable bins.
- Writes JSON files:
  - `train_xxx.json`
  - `test_xxx.json`

### `TypeDatasetBuilder`
- Groups samples by target class/type.
- Ensures balanced number of samples per class.
- Writes JSON files:
  - `train_xxx.json`
  - `test_xxx.json`

---

## 🟪 **dataset.py**
### **Purpose:**  
Wraps datasets and defines the PyTorch `Dataset` implementation.

### **Responsibilities:**
- Defines `LOSDataset`:
  - Loads image tensors via PIL.
  - Converts labels to:
    - single-element tensor (regression)
    - one-hot vector (classification)
  - Applies preprocessing transform.
- Provides tuple outputs for DataLoader:

---

## 🟥 **config.py**
### **Purpose:**  
Holds project-wide configuration and constants.

### **Responsibilities:**
- Defines enums such as:
  - `LabelType.DISTANCE`
  - `LabelType.TYPE_CLASSIFICATION`
- Central place for:
  - learning rate
  - batch size
  - dataset size
  - training split
  - model hyperparameters
- Intended to prevent scattering magic numbers across modules.

---

## ⬜️**features.py** 
### **Purpose:**  
This module implements a simple on-disk cache for precomputed DINO features.

### **Responsibilities:**
- `FeatureStore` class manages saving/loading `.pt` tensors.
- Each tensor corresponds to one image (indexed by its stem path).
- Features are always saved on CPU for portability.
- Allows the training pipeline to skip DINO computation if cached features exist.
- Provides methods: `exists()`, `save()`, `load()`, `feature_path()`.

---

## 🟫**precompute.py** 
### **Purpose:** 
This module performs bulk precomputation of DINO features for a dataset.

### **Responsibilities:**
- `precompute_features(model, dataset, store, device)` iterates through dataset items.
- Loads each image once, runs `model.encode(image)` to get DINO features.
- Stores the features to disk via `FeatureStore`.
- Skips samples whose features are already cached.
- Prevents DINO from being executed inside DataLoader workers.
- Makes training faster and more stable by using cached tensors.

---