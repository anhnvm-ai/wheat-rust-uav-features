"""
================================================================================
WHEAT DISEASE CLASSIFICATION - RAW DATA V7 (NO PCA)
Beyond Visible Spectrum: AI for Agriculture 2026
================================================================================

APPROACH: Direct raw data processing WITHOUT PCA
- Mean/Std per band (không mất thông tin spectral)
- Spatial statistics (texture, gradients)
- All vegetation indices từ raw bands
================================================================================
"""

import os
import numpy as np
import pandas as pd
import cv2
import tifffile
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIGURATION ==========================
DATA_PATH = r"D:\Kaggle2026\paper\beyond-visible-spectrum-ai-for-agriculture-2026\Kaggle_Prepared"
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VAL_PATH = os.path.join(DATA_PATH, "val")
OUTPUT_PATH = r"D:\Kaggle2026\paper\output_v7"

os.makedirs(OUTPUT_PATH, exist_ok=True)

CLASSES = ['Health', 'Rust', 'Other']
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ========================== DATA LOADING ==========================

def load_sample(sample_id, base_path):
    """Load all modalities for a sample"""
    rgb_path = os.path.join(base_path, "RGB", f"{sample_id}.png")
    ms_path = os.path.join(base_path, "MS", f"{sample_id}.tif")
    hs_path = os.path.join(base_path, "HS", f"{sample_id}.tif")
    
    rgb = cv2.imread(rgb_path)
    if rgb is not None:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    try:
        ms = tifffile.imread(ms_path)
        if len(ms.shape) == 3 and ms.shape[0] < ms.shape[2]:
            ms = np.moveaxis(ms, 0, -1)
    except:
        ms = None
    
    try:
        hs = tifffile.imread(hs_path)
        if len(hs.shape) == 3 and hs.shape[0] < hs.shape[2]:
            hs = np.moveaxis(hs, 0, -1)
    except:
        hs = None
    
    return rgb, ms, hs

def get_sample_id(filename):
    return os.path.splitext(filename)[0]

def get_class_label(filename):
    return filename.split('_')[0]

# ========================== RAW FEATURE EXTRACTION ==========================

def extract_rgb_raw_features(rgb):
    """Extract features from raw RGB data"""
    if rgb is None:
        return np.zeros(30)
    
    features = []
    rgb = rgb.astype(np.float32)
    
    # Per-channel: mean, std, min, max, percentiles
    for c in range(3):
        ch = rgb[:, :, c]
        features.extend([
            np.mean(ch),
            np.std(ch),
            np.min(ch),
            np.max(ch),
            np.percentile(ch, 25),
            np.percentile(ch, 75)
        ])
    
    # Global color features
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    # ExG (Excess Green) - healthy vegetation
    exg = 2*np.mean(g) - np.mean(r) - np.mean(b)
    features.append(exg / 255.0)
    
    # ExR (Excess Red) - rust indicator
    exr = 1.4*np.mean(r) - np.mean(g)
    features.append(exr / 255.0)
    
    # Color ratios
    mr, mg, mb = np.mean(r), np.mean(g), np.mean(b)
    features.append(mr / (mg + 1e-8))
    features.append(mr / (mb + 1e-8))
    features.append(mg / (mb + 1e-8))
    
    # Normalized differences
    features.append((mr - mg) / (mr + mg + 1e-8))
    features.append((mg - mb) / (mg + mb + 1e-8))
    
    # Texture: gradient magnitude
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    features.extend([np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)])
    
    # HSV statistics
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    features.extend([np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])])  # Hue
    features.extend([np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])])  # Saturation
    
    return np.array(features[:30])

def extract_ms_raw_features(ms):
    """Extract features from raw Multispectral data"""
    if ms is None:
        return np.zeros(50)
    
    features = []
    ms = ms.astype(np.float32)
    n_bands = ms.shape[-1]
    
    # Per-band statistics
    for b in range(min(n_bands, 5)):
        band = ms[:, :, b]
        features.extend([
            np.mean(band),
            np.std(band),
            np.min(band),
            np.max(band),
            np.percentile(band, 10),
            np.percentile(band, 90)
        ])
    
    # Vegetation indices (if 5 bands: Blue, Green, Red, RedEdge, NIR)
    if n_bands >= 5:
        blue = np.mean(ms[:,:,0])
        green = np.mean(ms[:,:,1])
        red = np.mean(ms[:,:,2])
        re = np.mean(ms[:,:,3])
        nir = np.mean(ms[:,:,4])
        
        # Core indices
        ndvi = (nir - red) / (nir + red + 1e-8)
        gndvi = (nir - green) / (nir + green + 1e-8)
        ndre = (nir - re) / (nir + re + 1e-8)
        sr = nir / (red + 1e-8)
        
        # Chlorophyll indices
        mtci = (nir - re) / (re - red + 1e-8)
        ci_green = nir / (green + 1e-8) - 1
        ci_re = nir / (re + 1e-8) - 1
        
        # Stress indices
        pri = (green - red) / (green + red + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        # Enhanced indices
        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8)
        savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        
        # Ratios
        nir_red = nir / (red + 1e-8)
        nir_green = nir / (green + 1e-8)
        re_red = re / (red + 1e-8)
        
        features.extend([
            ndvi, gndvi, ndre, sr,
            mtci, ci_green, ci_re,
            pri, ndwi, evi, savi,
            nir_red, nir_green, re_red
        ])
        
        # Spatial variability per band
        for b in range(5):
            band = ms[:, :, b]
            cv = np.std(band) / (np.mean(band) + 1e-8)
            features.append(cv)
    else:
        features.extend([0] * 19)
    
    return np.array(features[:50])

def extract_hs_raw_features(hs):
    """Extract features from raw Hyperspectral data"""
    if hs is None:
        return np.zeros(70)
    
    features = []
    hs = hs.astype(np.float32)
    n_bands = hs.shape[-1]
    
    # Mean spectrum across all pixels
    mean_spectrum = np.mean(hs, axis=(0, 1))
    std_spectrum = np.std(hs, axis=(0, 1))
    
    # Remove noisy edge bands
    if n_bands > 24:
        clean_start, clean_end = 10, n_bands - 14
        mean_clean = mean_spectrum[clean_start:clean_end]
        std_clean = std_spectrum[clean_start:clean_end]
    else:
        mean_clean = mean_spectrum
        std_clean = std_spectrum
    
    n_clean = len(mean_clean)
    
    # Sample spectrum at 20 evenly spaced positions
    n_samples = 20
    if n_clean >= n_samples:
        indices = np.linspace(0, n_clean-1, n_samples).astype(int)
        features.extend(mean_clean[indices] / 5000.0)  # Normalize
    else:
        features.extend(list(mean_clean / 5000.0) + [0]*(n_samples - n_clean))
    
    # Global spectral statistics
    features.extend([
        np.mean(mean_clean) / 5000.0,
        np.std(mean_clean) / 5000.0,
        np.min(mean_clean) / 5000.0,
        np.max(mean_clean) / 5000.0,
        (np.max(mean_clean) - np.min(mean_clean)) / 5000.0,
        np.median(mean_clean) / 5000.0
    ])
    
    # Spectral shape features
    if n_clean > 1:
        peak_pos = np.argmax(mean_clean) / n_clean
        valley_pos = np.argmin(mean_clean) / n_clean
        features.extend([peak_pos, valley_pos])
    else:
        features.extend([0, 0])
    
    # First derivative (spectral slope)
    if n_clean > 2:
        deriv = np.gradient(mean_clean)
        features.extend([
            np.mean(deriv) / 5000.0,
            np.std(deriv) / 5000.0,
            np.max(deriv) / 5000.0,
            np.argmax(deriv) / n_clean  # Position of max slope (red edge)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Red edge position and slope
    if n_clean >= 50:
        # Red edge region (roughly 35-50% of spectrum)
        re_start = int(n_clean * 0.35)
        re_end = int(n_clean * 0.50)
        re_region = mean_clean[re_start:re_end]
        if len(re_region) > 1:
            re_deriv = np.gradient(re_region)
            reip = (re_start + np.argmax(re_deriv)) / n_clean
            re_slope = np.max(re_deriv) / 5000.0
            features.extend([reip, re_slope])
        else:
            features.extend([0, 0])
        
        # Chlorophyll absorption (~30% of spectrum)
        chlor_idx = int(n_clean * 0.30)
        nir_idx = int(n_clean * 0.65)
        if chlor_idx < n_clean and nir_idx < n_clean:
            chlor_val = mean_clean[chlor_idx]
            nir_val = mean_clean[nir_idx]
            hs_ndvi = (nir_val - chlor_val) / (nir_val + chlor_val + 1e-8)
            features.append(hs_ndvi)
        else:
            features.append(0)
    else:
        features.extend([0, 0, 0])
    
    # Band-to-band correlation (smoothness)
    if n_clean > 10:
        correlations = []
        for i in range(0, n_clean-10, 10):
            corr = np.corrcoef(mean_clean[i:i+5], mean_clean[i+5:i+10])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        if correlations:
            features.append(np.mean(correlations))
        else:
            features.append(0)
    else:
        features.append(0)
    
    # Area under curve (different regions)
    if n_clean >= 6:
        total = np.trapz(mean_clean)
        third1 = np.trapz(mean_clean[:n_clean//3])
        third2 = np.trapz(mean_clean[n_clean//3:2*n_clean//3])
        third3 = np.trapz(mean_clean[2*n_clean//3:])
        features.extend([
            third1 / (total + 1e-8),
            third2 / (total + 1e-8),
            third3 / (total + 1e-8)
        ])
    else:
        features.extend([0, 0, 0])
    
    # Spatial variability (texture in spectral domain)
    if n_clean >= 10:
        sample_indices = np.linspace(0, n_clean-1, 10).astype(int)
        for idx in sample_indices[:5]:
            if idx < hs.shape[-1]:
                band = hs[:, :, min(idx + clean_start if 'clean_start' in dir() else idx, hs.shape[-1]-1)]
                cv = np.std(band) / (np.mean(band) + 1e-8)
                features.append(cv)
            else:
                features.append(0)
    else:
        features.extend([0] * 5)
    
    # Pad to 70 features
    while len(features) < 70:
        features.append(0)
    
    return np.array(features[:70])

def extract_all_features(sample_id, base_path):
    """Extract all features from raw data"""
    rgb, ms, hs = load_sample(sample_id, base_path)
    
    rgb_feats = extract_rgb_raw_features(rgb)
    ms_feats = extract_ms_raw_features(ms)
    hs_feats = extract_hs_raw_features(hs)
    
    return np.concatenate([rgb_feats, ms_feats, hs_feats])

# ========================== BUILD DATASET ==========================

def build_dataset(path, is_val=False):
    """Build dataset from raw features"""
    print(f"\nBuilding {'validation' if is_val else 'training'} dataset...")
    
    rgb_files = os.listdir(os.path.join(path, "RGB"))
    
    all_features = []
    all_labels = []
    all_ids = []
    
    for filename in tqdm(rgb_files):
        sample_id = get_sample_id(filename)
        
        if not is_val:
            label = get_class_label(filename)
            if label not in CLASSES:
                continue
            all_labels.append(label)
        
        features = extract_all_features(sample_id, path)
        all_features.append(features)
        all_ids.append(sample_id)
    
    X = np.vstack(all_features)
    print(f"  Shape: {X.shape}")
    
    return X, all_labels if not is_val else None, all_ids

# ========================== MODEL TRAINING ==========================

def train_models(X_train, y_train, X_val):
    """Train models on raw features"""
    print("\n" + "="*60)
    print("TRAINING MODELS (Raw Features, No PCA)")
    print("="*60)
    
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle NaN/Inf
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)
    
    # Models with regularization
    models = {
        'lgb': lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=20,
            max_depth=5,
            min_child_samples=15,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        'xgb': XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=8,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=RANDOM_STATE
        )
    }
    
    # Cross-validation
    print("\nCross-validation (5-fold):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores = {}
    trained_models = {}
    
    for name, model in models.items():
        y_pred_cv = cross_val_predict(model, X_train_scaled, y_train_enc, cv=cv)
        score = accuracy_score(y_train_enc, y_pred_cv)
        cv_scores[name] = score
        print(f"  {name}: {score:.4f}")
        
        model.fit(X_train_scaled, y_train_enc)
        trained_models[name] = model
    
    # Ensemble
    print("\nCreating weighted ensemble...")
    all_probas = []
    for name, model in trained_models.items():
        proba = model.predict_proba(X_val_scaled)
        weight = cv_scores[name]
        all_probas.append(proba * weight)
    
    total_weight = sum(cv_scores.values())
    avg_proba = np.sum(all_probas, axis=0) / total_weight
    y_pred = np.argmax(avg_proba, axis=1)
    
    avg_cv = np.mean(list(cv_scores.values()))
    best_model = max(cv_scores, key=cv_scores.get)
    print(f"\nAverage CV: {avg_cv:.4f}")
    print(f"Best: {best_model} ({cv_scores[best_model]:.4f})")
    
    return y_pred, le, scaler, trained_models, cv_scores

# ========================== MAIN ==========================

def main():
    print("="*70)
    print("WHEAT DISEASE CLASSIFICATION - RAW DATA V7 (NO PCA)")
    print("="*70)
    print("\n150 features: RGB(30) + MS(50) + HS(70)")
    
    # Build datasets
    X_train, y_train, train_ids = build_dataset(TRAIN_PATH, is_val=False)
    X_val, _, val_ids = build_dataset(VAL_PATH, is_val=True)
    
    print(f"\nTraining: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    
    # Train
    y_pred, le, scaler, models, cv_scores = train_models(X_train, y_train, X_val)
    y_pred_labels = le.inverse_transform(y_pred)
    
    # Submission
    submission = pd.DataFrame({
        'Id': [f"{sid}.tif" for sid in val_ids],
        'Category': y_pred_labels
    })
    
    submission.to_csv(os.path.join(OUTPUT_PATH, 'submission_v7.csv'), index=False)
    submission.to_csv(r'D:\Kaggle2026\paper\submission_v7.csv', index=False)
    
    print("\n" + "="*60)
    print("SUBMISSION CREATED")
    print("="*60)
    print(f"\nFile: D:\\Kaggle2026\\paper\\submission_v7.csv")
    print(f"\nDistribution:")
    print(submission['Category'].value_counts())
    
    # Save
    joblib.dump({
        'scaler': scaler,
        'models': models,
        'le': le,
        'cv_scores': cv_scores
    }, os.path.join(OUTPUT_PATH, 'ensemble_v7.pkl'))
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
