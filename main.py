import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

base_dir = Path('/kaggle/input/hsi-classification')
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'
labels = base_dir / 'labels.csv'

labels_df = pd.read_csv(labels)

X_train = []
y_train = []
for _, row in labels_df.iterrows():
    patch = np.load(train_dir / row['filename'])
    X_train.append(patch)
    y_train.append(row['label'])

X_train = np.array(X_train)
y_train = np.array(y_train)

class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
r_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
y_train = np.array([class_mapping[y] for y in y_train])

def normalize(X):
    X_norm = np.zeros_like(X, dtype=np.float32)
    n_bands = X.shape[3]
    
    for band in range(n_bands):
        band_data = X[:, :, :, band]
        band_min = band_data.min()
        band_max = band_data.max()
        
        if band_max > band_min:
            X_norm[:, :, :, band] = (band_data - band_min) / (band_max - band_min)
        else:
            X_norm[:, :, :, band] = 0
    
    return X_norm

X_train_norm = normalize(X_train)

# extrage peste 1000 de features din fiecare patch prin mai multe metode
# se extrage semnatura spectrala a pixelului central si se calculeaza statistici pe cele 48 benzi
# se calculeaza gradienti care arata cum se schimba valorile intre benzi consecutive
# se extrage context la mai multe scale folosind regiuni de 3x3 5x5 7x7 si 19x19 pixeli
# se calculeaza contrast intre centru si imprejurimi prin diferente si rapoarte
# se extrag gradienti pe orizontala verticala si diagonale pentru a captura textura si margini
# se calculeaza diferente fata de colturile patch ului pentru contrast periferic
# se calculeaza rapoarte spectrale intre benzi joase medii si inalte

def extract(X):
    N = X.shape[0]
    features = []
    
    for i in range(N):
        patch = X[i]
        f = []
        
        center = patch[9, 9, :]
        f.extend(center)
        f.append(center.mean())
        f.append(center.std())
        f.append(center.min())
        f.append(center.max())
        f.append(np.median(center))
        f.append(skew(center))
        f.append(kurtosis(center))
        
        grad1 = np.diff(center)
        grad2 = np.diff(grad1)
        f.extend(grad1)
        f.extend(grad2)
        f.append(np.abs(grad1).mean())
        f.append(np.abs(grad2).mean())
        
        r3 = patch[8:11, 8:11, :]
        mean_3x3 = r3.mean(axis=(0, 1))
        std_3x3 = r3.std(axis=(0, 1))
        f.extend(mean_3x3)
        f.extend(std_3x3)
        
        r5 = patch[7:12, 7:12, :]
        mean_5x5 = r5.mean(axis=(0, 1))
        std_5x5 = r5.std(axis=(0, 1))
        f.extend(mean_5x5)
        f.extend(std_5x5)
        
        r7 = patch[6:13, 6:13, :]
        mean_7x7 = r7.mean(axis=(0, 1))
        std_7x7 = r7.std(axis=(0, 1))
        f.extend(mean_7x7)
        f.extend(std_7x7)
        
        mean_full = patch.mean(axis=(0, 1))
        std_full = patch.std(axis=(0, 1))
        f.extend(mean_full)
        f.extend(std_full)
        
        f.extend(center - mean_3x3)
        f.extend(center - mean_5x5)
        f.extend(center - mean_7x7)
        f.extend(center / (mean_3x3 + 1e-6))
        f.extend(mean_3x3 / (mean_5x5 + 1e-6))
        
        grad_h = patch[9, 10, :] - patch[9, 8, :]
        grad_v = patch[10, 9, :] - patch[8, 9, :]
        f.extend(grad_h)
        f.extend(grad_v)
        f.extend(np.abs(grad_h) + np.abs(grad_v))
        
        grad_d1 = patch[10, 10, :] - patch[8, 8, :]
        grad_d2 = patch[10, 8, :] - patch[8, 10, :]
        f.extend(grad_d1)
        f.extend(grad_d2)
        
        corners = [patch[0, 0, :], patch[0, 18, :], patch[18, 0, :], patch[18, 18, :]]
        corner_mean = np.mean(corners, axis=0)
        f.extend(center - corner_mean)
        
        low_bands = center[:16].mean()
        mid_bands = center[16:32].mean()
        high_bands = center[32:].mean()
        f.append(low_bands / (mid_bands + 1e-6))
        f.append(mid_bands / (high_bands + 1e-6))
        f.append(low_bands / (high_bands + 1e-6))
        f.append((mid_bands - low_bands) / (mid_bands + low_bands + 1e-6))
        f.append((high_bands - mid_bands) / (high_bands + mid_bands + 1e-6))
        
        features.append(f)
    
    return np.array(features, dtype=np.float32)

X_features = extract(X_train_norm)
X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

pca = PCA(n_components=0.98, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_combined = np.hstack([X_scaled, X_pca])

counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)

weights = {}
for c, count in counts.items():
    base_weight = total / (n_classes * count)
    
    if c == 3:
        weights[c] = base_weight * 0.05
    elif c == 4:
        weights[c] = base_weight * 0.0085
    else:
        weights[c] = base_weight

sample_weights = np.array([weights[y] for y in y_train])

lgb_model = LGBMClassifier(
    n_estimators=900,
    max_depth=10,
    learning_rate=0.025,
    subsample=0.75,
    colsample_bytree=0.65,
    min_child_weight=3,
    reg_alpha=0.15,
    reg_lambda=1.5,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_combined, y_train, sample_weight=sample_weights)

test_files = sorted(test_dir.glob('*.npy'))
X_test = np.array([np.load(f) for f in test_files])
test_filenames = [f.name for f in test_files]

X_test_norm = normalize(X_test)

X_test_features = extract(X_test_norm)
X_test_features = np.nan_to_num(X_test_features, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = scaler.transform(X_test_features)
X_test_pca = pca.transform(X_test_scaled)
X_test_combined = np.hstack([X_test_scaled, X_test_pca])

lgb_predictions = lgb_model.predict(X_test_combined)
lgb_predictions_orig = np.array([r_mapping[p] for p in lgb_predictions])

submission = pd.DataFrame({
    'filename': test_filenames,
    'label': lgb_predictions_orig
})
submission.to_csv('/kaggle/working/submission.csv', index=Falseimport numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

base_dir = Path('/kaggle/input/hsi-classification')
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'
labels = base_dir / 'labels.csv'

labels_df = pd.read_csv(labels)

X_train = []
y_train = []
for _, row in labels_df.iterrows():
    patch = np.load(train_dir / row['filename'])
    X_train.append(patch)
    y_train.append(row['label'])

X_train = np.array(X_train)
y_train = np.array(y_train)

class_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
r_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
y_train = np.array([class_mapping[y] for y in y_train])

def normalize(X):
    X_norm = np.zeros_like(X, dtype=np.float32)
    n_bands = X.shape[3]
    
    for band in range(n_bands):
        band_data = X[:, :, :, band]
        band_min = band_data.min()
        band_max = band_data.max()
        
        if band_max > band_min:
            X_norm[:, :, :, band] = (band_data - band_min) / (band_max - band_min)
        else:
            X_norm[:, :, :, band] = 0
    
    return X_norm

X_train_norm = normalize(X_train)

# extrage peste 1000 de features din fiecare patch prin mai multe metode
# se extrage semnatura spectrala a pixelului central si se calculeaza statistici pe cele 48 benzi
# se calculeaza gradienti care arata cum se schimba valorile intre benzi consecutive
# se extrage context la mai multe scale folosind regiuni de 3x3 5x5 7x7 si 19x19 pixeli
# se calculeaza contrast intre centru si imprejurimi prin diferente si rapoarte
# se extrag gradienti pe orizontala verticala si diagonale pentru a captura textura si margini
# se calculeaza diferente fata de colturile patch ului pentru contrast periferic
# se calculeaza rapoarte spectrale intre benzi joase medii si inalte

def extract(X):
    N = X.shape[0]
    features = []
    
    for i in range(N):
        patch = X[i]
        f = []
        
        center = patch[9, 9, :]
        f.extend(center)
        f.append(center.mean())
        f.append(center.std())
        f.append(center.min())
        f.append(center.max())
        f.append(np.median(center))
        f.append(skew(center))
        f.append(kurtosis(center))
        
        grad1 = np.diff(center)
        grad2 = np.diff(grad1)
        f.extend(grad1)
        f.extend(grad2)
        f.append(np.abs(grad1).mean())
        f.append(np.abs(grad2).mean())
        
        r3 = patch[8:11, 8:11, :]
        mean_3x3 = r3.mean(axis=(0, 1))
        std_3x3 = r3.std(axis=(0, 1))
        f.extend(mean_3x3)
        f.extend(std_3x3)
        
        r5 = patch[7:12, 7:12, :]
        mean_5x5 = r5.mean(axis=(0, 1))
        std_5x5 = r5.std(axis=(0, 1))
        f.extend(mean_5x5)
        f.extend(std_5x5)
        
        r7 = patch[6:13, 6:13, :]
        mean_7x7 = r7.mean(axis=(0, 1))
        std_7x7 = r7.std(axis=(0, 1))
        f.extend(mean_7x7)
        f.extend(std_7x7)
        
        mean_full = patch.mean(axis=(0, 1))
        std_full = patch.std(axis=(0, 1))
        f.extend(mean_full)
        f.extend(std_full)
        
        f.extend(center - mean_3x3)
        f.extend(center - mean_5x5)
        f.extend(center - mean_7x7)
        f.extend(center / (mean_3x3 + 1e-6))
        f.extend(mean_3x3 / (mean_5x5 + 1e-6))
        
        grad_h = patch[9, 10, :] - patch[9, 8, :]
        grad_v = patch[10, 9, :] - patch[8, 9, :]
        f.extend(grad_h)
        f.extend(grad_v)
        f.extend(np.abs(grad_h) + np.abs(grad_v))
        
        grad_d1 = patch[10, 10, :] - patch[8, 8, :]
        grad_d2 = patch[10, 8, :] - patch[8, 10, :]
        f.extend(grad_d1)
        f.extend(grad_d2)
        
        corners = [patch[0, 0, :], patch[0, 18, :], patch[18, 0, :], patch[18, 18, :]]
        corner_mean = np.mean(corners, axis=0)
        f.extend(center - corner_mean)
        
        low_bands = center[:16].mean()
        mid_bands = center[16:32].mean()
        high_bands = center[32:].mean()
        f.append(low_bands / (mid_bands + 1e-6))
        f.append(mid_bands / (high_bands + 1e-6))
        f.append(low_bands / (high_bands + 1e-6))
        f.append((mid_bands - low_bands) / (mid_bands + low_bands + 1e-6))
        f.append((high_bands - mid_bands) / (high_bands + mid_bands + 1e-6))
        
        features.append(f)
    
    return np.array(features, dtype=np.float32)

X_features = extract(X_train_norm)
X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

pca = PCA(n_components=0.98, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_combined = np.hstack([X_scaled, X_pca])

counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)

weights = {}
for c, count in counts.items():
    base_weight = total / (n_classes * count)
    
    if c == 3:
        weights[c] = base_weight * 0.05
    elif c == 4:
        weights[c] = base_weight * 0.0085
    else:
        weights[c] = base_weight

sample_weights = np.array([weights[y] for y in y_train])

lgb_model = LGBMClassifier(
    n_estimators=900,
    max_depth=10,
    learning_rate=0.025,
    subsample=0.75,
    colsample_bytree=0.65,
    min_child_weight=3,
    reg_alpha=0.15,
    reg_lambda=1.5,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_combined, y_train, sample_weight=sample_weights)

test_files = sorted(test_dir.glob('*.npy'))
X_test = np.array([np.load(f) for f in test_files])
test_filenames = [f.name for f in test_files]

X_test_norm = normalize(X_test)

X_test_features = extract(X_test_norm)
X_test_features = np.nan_to_num(X_test_features, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = scaler.transform(X_test_features)
X_test_pca = pca.transform(X_test_scaled)
X_test_combined = np.hstack([X_test_scaled, X_test_pca])

lgb_predictions = lgb_model.predict(X_test_combined)
lgb_predictions_orig = np.array([r_mapping[p] for p in lgb_predictions])

submission = pd.DataFrame({
    'filename': test_filenames,
    'label': lgb_predictions_orig
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
