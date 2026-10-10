"""
Configuration file for Beyond Visible Spectrum: AI for Agriculture Project
==========================================================================
Cấu hình tập trung cho toàn bộ project
"""

import os
from pathlib import Path

# ==================== PATHS ====================
ROOT_DIR = Path("/content/drive/MyDrive/Kaggle_Prepared")
TRAIN_DIR = ROOT_DIR / "train"
VAL_DIR = ROOT_DIR / "val"

# Modality paths
TRAIN_RGB_DIR = TRAIN_DIR / "RGB"
TRAIN_MS_DIR = TRAIN_DIR / "MS"
TRAIN_HS_DIR = TRAIN_DIR / "HS"

VAL_RGB_DIR = VAL_DIR / "RGB"
VAL_MS_DIR = VAL_DIR / "MS"
VAL_HS_DIR = VAL_DIR / "HS"

# Output directories
OUTPUT_DIR = ROOT_DIR / "outputs"
EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if not exist
for dir_path in [OUTPUT_DIR, EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== DATA CONFIG ====================
# Class mapping (based on actual file prefixes: Health, Other, Rust)
CLASS_NAMES = ['Health', 'Other', 'Rust']
CLASS_TO_IDX = {'Health': 0, 'Other': 1, 'Rust': 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

# Modality specifications
RGB_CHANNELS = 3
MS_BANDS = 5  # Multispectral bands
HS_TOTAL_BANDS = 125  # Total hyperspectral bands
HS_PRUNE_START = 10  # Remove first 10 noisy bands
HS_PRUNE_END = 14    # Remove last 14 noisy bands
HS_BANDS = HS_TOTAL_BANDS - HS_PRUNE_START - HS_PRUNE_END  # 101 useful bands

# Image dimensions
IMG_SIZE = 224  # For ResNet input
HS_SPATIAL_SIZE = 64  # HS images spatial dimension

# Wavelength range for HS (approximate)
HS_WAVELENGTH_START = 400  # nm
HS_WAVELENGTH_END = 1000   # nm

# ==================== TRAINING CONFIG ====================
# General
RANDOM_SEED = 42
NUM_WORKERS = 2

# K-Fold
N_FOLDS = 5

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# ==================== MODEL CONFIG ====================
# Feature dimensions after backbone
RGB_FEATURE_DIM = 512   # ResNet18 output
MS_FEATURE_DIM = 256    # Custom CNN output
HS_FEATURE_DIM = 128    # Conv1D output
FUSION_DIM = RGB_FEATURE_DIM + MS_FEATURE_DIM + HS_FEATURE_DIM

# Dropout rates
DROPOUT_RATE = 0.5

# ==================== VEGETATION INDICES ====================
# MS band indices (0-indexed)
MS_BAND_NAMES = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
MS_BLUE_IDX = 0
MS_GREEN_IDX = 1
MS_RED_IDX = 2
MS_REDEDGE_IDX = 3
MS_NIR_IDX = 4

# ==================== METRICS CONFIG ====================
METRICS_TO_COMPUTE = ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall', 'roc_auc']

# ==================== LITERATURE BASELINES ====================
# Reference results from papers for comparison
LITERATURE_BASELINES = {
    'Guo_2023_DCNN': {'accuracy': 0.85, 'f1_macro': 0.84},
    'Guo_2023_RF': {'accuracy': 0.78, 'f1_macro': 0.77},
    'Guo_2025_ViT': {'accuracy': 0.92, 'f1_macro': 0.91},
    'Guo_2025_Fusion': {'accuracy': 0.95, 'f1_macro': 0.94},
}

# ==================== DEVICE CONFIG ====================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Class Names: {CLASS_NAMES}")
    print(f"\nHS Bands (after pruning): {HS_BANDS}")
    print(f"MS Bands: {MS_BANDS}")
    print(f"RGB Channels: {RGB_CHANNELS}")
    print(f"\nBatch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"K-Folds: {N_FOLDS}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
