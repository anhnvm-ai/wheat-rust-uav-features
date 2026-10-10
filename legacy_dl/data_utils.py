"""
Data Utilities for Beyond Visible Spectrum Project
===================================================
Các hàm tiện ích để load, preprocess và validate data
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Image processing
from PIL import Image
import rasterio
from rasterio.errors import RasterioIOError

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Progress bar
from tqdm import tqdm

# Local imports
from config import *


def load_rgb_image(filepath: Union[str, Path], resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load RGB image from .png file
    
    Args:
        filepath: Path to RGB image
        resize: Optional tuple (height, width) to resize
        
    Returns:
        numpy array of shape (H, W, 3) with values in [0, 255]
    """
    img = Image.open(filepath).convert('RGB')
    if resize:
        img = img.resize((resize[1], resize[0]), Image.BILINEAR)
    return np.array(img)


def load_ms_image(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load Multispectral image from .tif file
    
    Args:
        filepath: Path to MS image
        
    Returns:
        numpy array of shape (bands, H, W)
    """
    with rasterio.open(filepath) as src:
        data = src.read()  # Shape: (bands, H, W)
    return data.astype(np.float32)


def load_hs_image(filepath: Union[str, Path], prune: bool = True) -> np.ndarray:
    """
    Load Hyperspectral image from .tif file
    
    Args:
        filepath: Path to HS image
        prune: Whether to remove noisy bands (first 10, last 14)
        
    Returns:
        numpy array of shape (bands, H, W)
    """
    with rasterio.open(filepath) as src:
        data = src.read()  # Shape: (bands, H, W)
    
    if prune:
        # Remove noisy bands
        data = data[HS_PRUNE_START:HS_TOTAL_BANDS - HS_PRUNE_END]
    
    return data.astype(np.float32)


def get_sample_id_from_filename(filename: str) -> str:
    """Extract sample ID from filename (remove extension)"""
    return Path(filename).stem


def get_class_from_filename(filename: str) -> Optional[str]:
    """
    Extract class label from filename
    Assumes format: ClassX_SampleID.ext (e.g., Healthy_001.png)
    """
    name = Path(filename).stem
    for class_name in CLASS_NAMES:
        if name.startswith(class_name):
            return class_name
    return None


def count_files_per_modality(root_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Count files in each modality folder
    
    Returns:
        Dictionary with counts per modality and split
    """
    counts = {}
    for split in ['train', 'val']:
        counts[split] = {}
        split_dir = root_dir / split
        for modality in ['RGB', 'MS', 'HS']:
            mod_dir = split_dir / modality
            if mod_dir.exists():
                counts[split][modality] = len(list(mod_dir.iterdir()))
            else:
                counts[split][modality] = 0
    return counts


def count_files_per_class(train_dir: Path, modality: str = 'RGB') -> Dict[str, int]:
    """
    Count files per class in training set
    
    Args:
        train_dir: Path to training directory
        modality: Which modality to count from
        
    Returns:
        Dictionary with counts per class
    """
    mod_dir = train_dir / modality
    counts = {cls: 0 for cls in CLASS_NAMES}
    
    for f in mod_dir.iterdir():
        cls = get_class_from_filename(f.name)
        if cls:
            counts[cls] += 1
    
    return counts


def validate_data_integrity(root_dir: Path) -> Dict:
    """
    Validate dataset integrity
    
    Checks:
    - All modalities have same number of samples
    - Filenames match across modalities
    - No corrupted files
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'counts': count_files_per_modality(root_dir),
        'missing_files': [],
        'corrupted_files': [],
        'mismatched_samples': []
    }
    
    for split in ['train', 'val']:
        split_dir = root_dir / split
        
        # Get all sample IDs from RGB
        rgb_dir = split_dir / 'RGB'
        if not rgb_dir.exists():
            results['valid'] = False
            continue
            
        rgb_samples = {get_sample_id_from_filename(f.name) for f in rgb_dir.iterdir() if f.suffix == '.png'}
        
        # Check MS and HS have matching samples
        for modality in ['MS', 'HS']:
            mod_dir = split_dir / modality
            if not mod_dir.exists():
                results['valid'] = False
                continue
                
            mod_samples = {get_sample_id_from_filename(f.name) for f in mod_dir.iterdir() if f.suffix == '.tif'}
            
            # Find missing
            missing_in_mod = rgb_samples - mod_samples
            if missing_in_mod:
                results['missing_files'].extend([f"{split}/{modality}/{s}" for s in missing_in_mod])
                results['valid'] = False
            
            missing_in_rgb = mod_samples - rgb_samples
            if missing_in_rgb:
                results['mismatched_samples'].extend([f"{split}/{modality}/{s}" for s in missing_in_rgb])
    
    # Test loading a few files
    try:
        rgb_test = list((root_dir / 'train' / 'RGB').iterdir())[0]
        load_rgb_image(rgb_test)
    except Exception as e:
        results['corrupted_files'].append(f"RGB load error: {e}")
        results['valid'] = False
    
    try:
        ms_test = list((root_dir / 'train' / 'MS').iterdir())[0]
        load_ms_image(ms_test)
    except Exception as e:
        results['corrupted_files'].append(f"MS load error: {e}")
        results['valid'] = False
    
    try:
        hs_test = list((root_dir / 'train' / 'HS').iterdir())[0]
        load_hs_image(hs_test)
    except Exception as e:
        results['corrupted_files'].append(f"HS load error: {e}")
        results['valid'] = False
    
    return results


def create_dataset_dataframe(root_dir: Path) -> pd.DataFrame:
    """
    Create a DataFrame with all samples and their paths
    
    Returns:
        DataFrame with columns: sample_id, class, rgb_path, ms_path, hs_path, split
    """
    records = []
    
    for split in ['train', 'val']:
        split_dir = root_dir / split
        rgb_dir = split_dir / 'RGB'
        
        if not rgb_dir.exists():
            continue
        
        for rgb_file in rgb_dir.iterdir():
            if rgb_file.suffix != '.png':
                continue
                
            sample_id = get_sample_id_from_filename(rgb_file.name)
            cls = get_class_from_filename(rgb_file.name)
            
            # Find corresponding MS and HS files
            ms_path = split_dir / 'MS' / f"{sample_id}.tif"
            hs_path = split_dir / 'HS' / f"{sample_id}.tif"
            
            records.append({
                'sample_id': sample_id,
                'class': cls,
                'class_idx': CLASS_TO_IDX.get(cls, -1) if cls else -1,
                'rgb_path': str(rgb_file),
                'ms_path': str(ms_path) if ms_path.exists() else None,
                'hs_path': str(hs_path) if hs_path.exists() else None,
                'split': split
            })
    
    return pd.DataFrame(records)


def extract_hs_features(hs_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract statistical features from hyperspectral data
    
    Args:
        hs_data: Hyperspectral cube of shape (bands, H, W)
        
    Returns:
        Dictionary with feature arrays
    """
    features = {}
    
    # Flatten spatial dimensions
    hs_flat = hs_data.reshape(hs_data.shape[0], -1)  # (bands, pixels)
    
    # Per-band statistics (across pixels)
    features['band_mean'] = np.mean(hs_flat, axis=1)  # (bands,)
    features['band_std'] = np.std(hs_flat, axis=1)
    features['band_min'] = np.min(hs_flat, axis=1)
    features['band_max'] = np.max(hs_flat, axis=1)
    features['band_median'] = np.median(hs_flat, axis=1)
    
    # Global statistics
    features['global_mean'] = np.mean(hs_data)
    features['global_std'] = np.std(hs_data)
    
    # Spectral curve (mean across spatial)
    features['spectral_curve'] = features['band_mean']
    
    return features


def compute_vegetation_indices(ms_data: np.ndarray) -> Dict[str, float]:
    """
    Compute vegetation indices from multispectral data
    
    Args:
        ms_data: MS image of shape (bands, H, W)
        
    Returns:
        Dictionary with index values (mean across image)
    """
    eps = 1e-8  # Avoid division by zero
    
    # Extract bands (assuming order: Blue, Green, Red, RedEdge, NIR)
    blue = ms_data[MS_BLUE_IDX].astype(np.float64)
    green = ms_data[MS_GREEN_IDX].astype(np.float64)
    red = ms_data[MS_RED_IDX].astype(np.float64)
    rededge = ms_data[MS_REDEDGE_IDX].astype(np.float64)
    nir = ms_data[MS_NIR_IDX].astype(np.float64)
    
    indices = {}
    
    # NDVI (Normalized Difference Vegetation Index)
    indices['NDVI'] = np.nanmean((nir - red) / (nir + red + eps))
    
    # NDRE (Normalized Difference Red Edge)
    indices['NDRE'] = np.nanmean((nir - rededge) / (nir + rededge + eps))
    
    # GNDVI (Green NDVI)
    indices['GNDVI'] = np.nanmean((nir - green) / (nir + green + eps))
    
    # EVI (Enhanced Vegetation Index) - simplified
    indices['EVI'] = np.nanmean(2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps))
    
    # SAVI (Soil Adjusted Vegetation Index)
    L = 0.5
    indices['SAVI'] = np.nanmean((nir - red) * (1 + L) / (nir + red + L + eps))
    
    # OSAVI (Optimized SAVI)
    indices['OSAVI'] = np.nanmean((nir - red) / (nir + red + 0.16 + eps))
    
    # CIgreen (Chlorophyll Index Green)
    indices['CIgreen'] = np.nanmean(nir / (green + eps) - 1)
    
    # CIrededge (Chlorophyll Index Red Edge)
    indices['CIrededge'] = np.nanmean(nir / (rededge + eps) - 1)
    
    return indices


def extract_all_features(rgb_path: str, ms_path: str, hs_path: str) -> Dict[str, float]:
    """
    Extract all features from a single sample
    
    Args:
        rgb_path, ms_path, hs_path: Paths to modality files
        
    Returns:
        Dictionary with all features as scalar values
    """
    features = {}
    
    # RGB features (basic statistics)
    try:
        rgb = load_rgb_image(rgb_path)
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'RGB_{channel}_mean'] = np.mean(rgb[:,:,i])
            features[f'RGB_{channel}_std'] = np.std(rgb[:,:,i])
    except Exception as e:
        print(f"Error loading RGB: {e}")
    
    # MS features (vegetation indices)
    try:
        if ms_path and Path(ms_path).exists():
            ms = load_ms_image(ms_path)
            vi = compute_vegetation_indices(ms)
            features.update(vi)
            
            # MS band statistics
            for i, band_name in enumerate(MS_BAND_NAMES):
                features[f'MS_{band_name}_mean'] = np.mean(ms[i])
                features[f'MS_{band_name}_std'] = np.std(ms[i])
    except Exception as e:
        print(f"Error loading MS: {e}")
    
    # HS features (spectral statistics)
    try:
        if hs_path and Path(hs_path).exists():
            hs = load_hs_image(hs_path, prune=True)
            hs_feats = extract_hs_features(hs)
            
            # Add summarized HS features
            features['HS_global_mean'] = hs_feats['global_mean']
            features['HS_global_std'] = hs_feats['global_std']
            
            # Add spectral curve statistics
            curve = hs_feats['spectral_curve']
            features['HS_curve_mean'] = np.mean(curve)
            features['HS_curve_std'] = np.std(curve)
            features['HS_curve_min'] = np.min(curve)
            features['HS_curve_max'] = np.max(curve)
            features['HS_curve_range'] = np.max(curve) - np.min(curve)
            
            # Add some band-specific features (key regions)
            n_bands = len(curve)
            features['HS_blue_region'] = np.mean(curve[:n_bands//5])
            features['HS_green_region'] = np.mean(curve[n_bands//5:2*n_bands//5])
            features['HS_red_region'] = np.mean(curve[2*n_bands//5:3*n_bands//5])
            features['HS_rededge_region'] = np.mean(curve[3*n_bands//5:4*n_bands//5])
            features['HS_nir_region'] = np.mean(curve[4*n_bands//5:])
    except Exception as e:
        print(f"Error loading HS: {e}")
    
    return features


def create_feature_matrix(df: pd.DataFrame, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create feature matrix from all samples
    
    Args:
        df: DataFrame with sample paths
        verbose: Whether to show progress bar
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        feature_names: List of feature names
    """
    all_features = []
    labels = []
    
    iterator = tqdm(df.iterrows(), total=len(df), desc="Extracting features") if verbose else df.iterrows()
    
    for idx, row in iterator:
        feats = extract_all_features(row['rgb_path'], row['ms_path'], row['hs_path'])
        all_features.append(feats)
        labels.append(row['class_idx'])
    
    # Convert to DataFrame for consistent feature ordering
    feature_df = pd.DataFrame(all_features)
    feature_names = list(feature_df.columns)
    
    # Handle missing values
    feature_df = feature_df.fillna(0)
    
    X = feature_df.values.astype(np.float32)
    y = np.array(labels)
    
    return X, y, feature_names


def get_kfold_splits(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS) -> List[Tuple]:
    """
    Generate stratified K-fold splits
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(X, y))
    return splits


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, 
                       method: str = 'standard') -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Normalize features using sklearn scalers
    
    Args:
        X_train, X_val: Feature matrices
        method: 'standard' (z-score) or 'minmax'
        
    Returns:
        X_train_norm, X_val_norm, scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    return X_train_norm, X_val_norm, scaler


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")
    
    # Validate data
    print("\n1. Validating data integrity...")
    validation = validate_data_integrity(ROOT_DIR)
    print(f"   Valid: {validation['valid']}")
    print(f"   Counts: {validation['counts']}")
    
    # Create dataset DataFrame
    print("\n2. Creating dataset DataFrame...")
    df = create_dataset_dataframe(ROOT_DIR)
    print(f"   Total samples: {len(df)}")
    print(f"   Train samples: {len(df[df['split'] == 'train'])}")
    print(f"   Val samples: {len(df[df['split'] == 'val'])}")
    
    # Test feature extraction on one sample
    print("\n3. Testing feature extraction...")
    sample = df.iloc[0]
    features = extract_all_features(sample['rgb_path'], sample['ms_path'], sample['hs_path'])
    print(f"   Number of features: {len(features)}")
    print(f"   Feature names: {list(features.keys())[:5]}...")
    
    print("\nData utilities test complete!")
