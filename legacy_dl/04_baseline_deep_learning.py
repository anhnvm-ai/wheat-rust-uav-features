"""
Step 4: Deep Learning Baseline Models
======================================
Xây dựng baseline DL với ResNet fusion model

Usage:
    python 04_baseline_deep_learning.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Data processing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# Progress
from tqdm import tqdm

# PIL for images
from PIL import Image
import rasterio

# Add parent directory
sys.path.append(str(Path(__file__).parent))

from config import *
from data_utils import create_dataset_dataframe, load_hs_image

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


# ==================== DATASETS ====================

class MultimodalDataset(Dataset):
    """Dataset for loading RGB, MS, and HS data"""
    
    def __init__(self, df: pd.DataFrame, transform_rgb=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.transform_rgb = transform_rgb
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load RGB
        rgb = self._load_rgb(row['rgb_path'])
        
        # Load MS
        ms = self._load_ms(row['ms_path'])
        
        # Load HS
        hs = self._load_hs(row['hs_path'])
        
        # Label
        label = row['class_idx']
        
        return {
            'rgb': rgb,
            'ms': ms,
            'hs': hs,
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': row['sample_id']
        }
    
    def _load_rgb(self, path):
        """Load and preprocess RGB image"""
        img = Image.open(path).convert('RGB')
        if self.transform_rgb:
            img = self.transform_rgb(img)
        else:
            img = transforms.ToTensor()(img)
        return img
    
    def _load_ms(self, path):
        """Load and preprocess MS image"""
        if path and Path(path).exists():
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
            # Normalize per band
            data = (data - data.mean(axis=(1, 2), keepdims=True)) / (data.std(axis=(1, 2), keepdims=True) + 1e-8)
            # Resize to match RGB
            data = torch.tensor(data)
            data = F.interpolate(data.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
            return data
        else:
            return torch.zeros(MS_BANDS, IMG_SIZE, IMG_SIZE)
    
    def _load_hs(self, path):
        """Load and preprocess HS image"""
        if path and Path(path).exists():
            hs = load_hs_image(path, prune=True)  # (bands, H, W)
            # Compute mean spectral curve
            spectral = hs.mean(axis=(1, 2))  # (bands,)
            # Normalize
            spectral = (spectral - spectral.mean()) / (spectral.std() + 1e-8)
            return torch.tensor(spectral, dtype=torch.float32)
        else:
            return torch.zeros(HS_BANDS)


class RGBOnlyDataset(Dataset):
    """Simple dataset for RGB only (for simpler baseline)"""
    
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img = Image.open(row['rgb_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(row['class_idx'], dtype=torch.long)
        
        return img, label


# ==================== MODELS ====================

class RGBModel(nn.Module):
    """ResNet18 for RGB images"""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.backbone = resnet18(weights=None)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features before final FC"""
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class MSModel(nn.Module):
    """CNN for Multispectral images"""
    
    def __init__(self, in_channels=MS_BANDS, num_classes=NUM_CLASSES):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class HSModel(nn.Module):
    """1D CNN for Hyperspectral spectral curves"""
    
    def __init__(self, in_channels=HS_BANDS, num_classes=NUM_CLASSES):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, bands)
        x = x.unsqueeze(1)  # (batch, 1, bands)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class FusionModel(nn.Module):
    """Late Fusion Model combining RGB, MS, and HS"""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained_rgb=True):
        super().__init__()
        
        # Individual encoders
        self.rgb_encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_rgb else None)
        self.rgb_encoder.fc = nn.Identity()  # Remove final FC
        
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(MS_BANDS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.hs_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Fusion dimensions: RGB(512) + MS(256) + HS(128) = 896
        fusion_dim = 512 + 256 + 128
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE / 2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, rgb, ms, hs):
        # Extract features from each modality
        rgb_feat = self.rgb_encoder(rgb)  # (batch, 512)
        ms_feat = self.ms_encoder(ms)      # (batch, 256)
        hs_feat = self.hs_encoder(hs.unsqueeze(1))  # (batch, 128)
        
        # Concatenate
        fused = torch.cat([rgb_feat, ms_feat, hs_feat], dim=1)
        
        # Classify
        out = self.fusion_classifier(fused)
        
        return out


# ==================== TRAINING FUNCTIONS ====================

def get_transforms(mode='train'):
    """Get data augmentation transforms"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device, model_type='rgb'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        if model_type == 'rgb':
            inputs, labels = batch
            inputs = inputs.to(device)
        elif model_type == 'fusion':
            inputs = {
                'rgb': batch['rgb'].to(device),
                'ms': batch['ms'].to(device),
                'hs': batch['hs'].to(device)
            }
            labels = batch['label']
        
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'rgb':
            outputs = model(inputs)
        elif model_type == 'fusion':
            outputs = model(inputs['rgb'], inputs['ms'], inputs['hs'])
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, model_type='rgb'):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            if model_type == 'rgb':
                inputs, labels = batch
                inputs = inputs.to(device)
            elif model_type == 'fusion':
                inputs = {
                    'rgb': batch['rgb'].to(device),
                    'ms': batch['ms'].to(device),
                    'hs': batch['hs'].to(device)
                }
                labels = batch['label']
            
            labels = labels.to(device)
            
            if model_type == 'rgb':
                outputs = model(inputs)
            elif model_type == 'fusion':
                outputs = model(inputs['rgb'], inputs['ms'], inputs['hs'])
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


def train_model(model, train_loader, val_loader, device, model_type='rgb', 
                epochs=EPOCHS, lr=LEARNING_RATE, model_name='model'):
    """Full training loop with early stopping"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print(f"\n  Training {model_name}...")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, LR: {lr}")
    
    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, model_type)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, device, model_type)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"    ✓ New best model! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n    Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc


def evaluate_model_kfold(df: pd.DataFrame, model_class, model_type='rgb', 
                          n_folds=N_FOLDS, epochs=EPOCHS):
    """Evaluate model with K-Fold cross validation"""
    
    train_df = df[df['split'] == 'train'].copy()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = {
        'accuracy': [], 'f1_macro': [], 'f1_weighted': [],
        'precision': [], 'recall': []
    }
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['class_idx'])):
        print(f"\n  === Fold {fold+1}/{n_folds} ===")
        
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Create datasets
        if model_type == 'rgb':
            train_dataset = RGBOnlyDataset(fold_train, transform=get_transforms('train'))
            val_dataset = RGBOnlyDataset(fold_val, transform=get_transforms('val'))
        else:
            train_dataset = MultimodalDataset(fold_train, transform_rgb=get_transforms('train'), mode='train')
            val_dataset = MultimodalDataset(fold_val, transform_rgb=get_transforms('val'), mode='val')
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        # Initialize model
        model = model_class()
        
        # Train
        model, history, best_acc = train_model(
            model, train_loader, val_loader, DEVICE,
            model_type=model_type, epochs=epochs, model_name=f'Fold {fold+1}'
        )
        
        # Final evaluation
        _, val_acc, val_f1, y_true, y_pred = validate_epoch(
            model, val_loader, nn.CrossEntropyLoss(), DEVICE, model_type
        )
        
        fold_results['accuracy'].append(val_acc)
        fold_results['f1_macro'].append(val_f1)
        fold_results['f1_weighted'].append(f1_score(y_true, y_pred, average='weighted'))
        fold_results['precision'].append(precision_score(y_true, y_pred, average='macro'))
        fold_results['recall'].append(recall_score(y_true, y_pred, average='macro'))
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Aggregate results
    results = {
        'accuracy_mean': np.mean(fold_results['accuracy']),
        'accuracy_std': np.std(fold_results['accuracy']),
        'f1_macro_mean': np.mean(fold_results['f1_macro']),
        'f1_macro_std': np.std(fold_results['f1_macro']),
        'f1_weighted_mean': np.mean(fold_results['f1_weighted']),
        'precision_mean': np.mean(fold_results['precision']),
        'recall_mean': np.mean(fold_results['recall']),
        'fold_results': fold_results,
        'all_y_true': np.array(all_y_true),
        'all_y_pred': np.array(all_y_pred)
    }
    
    return results


def plot_training_history(history: dict, save_path: Path, model_name: str):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2 = axes[1]
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / f'{model_name.lower().replace(" ", "_")}_history.png', dpi=150)
    plt.close()


def plot_dl_results(results: dict, save_path: Path):
    """Plot DL model results"""
    
    # Confusion matrices
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        cm = confusion_matrix(res['all_y_true'], res['all_y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name}\nAcc: {res["accuracy_mean"]:.3f}')
    
    plt.suptitle('Confusion Matrices - Deep Learning Baselines', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrices_dl.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comparison bar chart
    names = list(results.keys())
    acc_means = [results[n]['accuracy_mean'] for n in names]
    acc_stds = [results[n]['accuracy_std'] for n in names]
    f1_means = [results[n]['f1_macro_mean'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Accuracy', capsize=3)
    bars2 = ax.bar(x + width/2, f1_means, width, label='F1-Macro', capsize=3)
    
    ax.set_ylabel('Score')
    ax.set_title('Deep Learning Baseline Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='Target (90%)')
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison_dl.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: confusion_matrices_dl.png, model_comparison_dl.png")


def save_dl_results(results: dict, save_path: Path):
    """Save DL results"""
    
    # Summary table
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}",
            'F1-Macro': f"{res['f1_macro_mean']:.4f} ± {res['f1_macro_std']:.4f}",
            'F1-Weighted': f"{res['f1_weighted_mean']:.4f}",
            'Precision': f"{res['precision_mean']:.4f}",
            'Recall': f"{res['recall_mean']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path / 'dl_baseline_results.csv', index=False)
    
    # Detailed JSON
    results_json = {}
    for name, res in results.items():
        results_json[name] = {
            'accuracy_mean': float(res['accuracy_mean']),
            'accuracy_std': float(res['accuracy_std']),
            'f1_macro_mean': float(res['f1_macro_mean']),
            'f1_macro_std': float(res['f1_macro_std']),
            'classification_report': classification_report(
                res['all_y_true'], res['all_y_pred'],
                target_names=CLASS_NAMES, output_dict=True
            )
        }
    
    with open(save_path / 'dl_baseline_detailed.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"  ✓ Saved: dl_baseline_results.csv, dl_baseline_detailed.json")
    
    return summary_df


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print(" BEYOND VISIBLE SPECTRUM: AI FOR AGRICULTURE")
    print(" Deep Learning Baseline Models")
    print("=" * 70)
    print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"  Output Directory: {RESULTS_DIR}")
    
    # Load dataset
    print_header("LOADING DATASET")
    df = create_dataset_dataframe(ROOT_DIR)
    print(f"  Total samples: {len(df)}")
    print(f"  Training samples: {len(df[df['split'] == 'train'])}")
    
    # Results storage
    all_results = {}
    
    # 1. RGB Only Model (ResNet18)
    print_header("MODEL 1: ResNet18 (RGB Only)")
    try:
        rgb_results = evaluate_model_kfold(
            df, RGBModel, model_type='rgb', 
            n_folds=N_FOLDS, epochs=min(EPOCHS, 30)
        )
        all_results['ResNet18 (RGB)'] = rgb_results
        print(f"\n  Results: Acc={rgb_results['accuracy_mean']:.4f}, F1={rgb_results['f1_macro_mean']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 2. Fusion Model (RGB + MS + HS)
    print_header("MODEL 2: Fusion Model (RGB + MS + HS)")
    try:
        fusion_results = evaluate_model_kfold(
            df, FusionModel, model_type='fusion',
            n_folds=N_FOLDS, epochs=min(EPOCHS, 30)
        )
        all_results['Fusion (RGB+MS+HS)'] = fusion_results
        print(f"\n  Results: Acc={fusion_results['accuracy_mean']:.4f}, F1={fusion_results['f1_macro_mean']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Visualizations
    if all_results:
        print_header("GENERATING VISUALIZATIONS")
        plot_dl_results(all_results, RESULTS_DIR)
        
        # Save results
        print_header("SAVING RESULTS")
        summary_df = save_dl_results(all_results, RESULTS_DIR)
        
        # Print summary
        print_header("RESULTS SUMMARY")
        print("\n" + summary_df.to_string(index=False))
        
        # Best model
        best_model = max(all_results.items(), key=lambda x: x[1]['accuracy_mean'])
        print(f"\n  🏆 Best DL Model: {best_model[0]}")
        print(f"     Accuracy: {best_model[1]['accuracy_mean']:.4f}")
        print(f"     F1-Macro: {best_model[1]['f1_macro_mean']:.4f}")
    
    print_header("DEEP LEARNING BASELINE COMPLETE")
    print(f"\n  ✓ All results saved to: {RESULTS_DIR}")
    print(f"\n  Next step: Run '05_final_report.py' for Final Report")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
