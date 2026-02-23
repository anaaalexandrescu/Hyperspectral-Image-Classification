# Hyperspectral Image Classification Project

## 1. Overview
This project implements a comprehensive machine learning pipeline for classifying hyperspectral images (HSI). The system processes individual hyperspectral image patches, extracts extensive spatial and spectral features, reduces dimensionality, and uses a LightGBM classifier to accurately predict the terrain or material class of the central pixel.

## 2. Pipeline Overview
The workflow is divided into several key stages:
* Data Loading and Label Mapping
* Band-wise Normalization
* Advanced Feature Engineering
* Scaling and Dimensionality Reduction (PCA)
* Custom Weighted Training using LightGBM
* Inference and Predictions Generation

## 3. Key Features and Technical Implementation

### Data Processing and Normalization
Hyperspectral patches are loaded as 3D Numpy arrays (19x19 spatial dimensions with 48 spectral bands). 
* *Normalization*: A custom Min-Max scaling function normalizes the data on a per-band basis. This ensures that the distinct spectral reflectances across different wavelengths are scaled between 0 and 1, preventing bands with naturally high intensity from dominating the model.
* *Label Mapping*: Target labels (1 through 7) are mapped to 0-indexed values (0 to 6) to be compatible with LightGBM's multi-class requirements, and reversed during the final output generation.

### Feature Engineering
Instead of relying solely on raw pixel values, this project leverages a highly detailed, hand-crafted feature extraction approach. Over 1000 features are generated per patch to capture both the chemical (spectral) and physical (spatial) properties of the target:
* *Central Pixel Statistics*: Extracts the raw spectral signature of the central pixel, alongside statistical measures like mean, standard deviation, min, max, median, skewness, and kurtosis.
* *Spectral Derivatives*: Computes 1st and 2nd order gradients (differences) between consecutive spectral bands. This captures the shape of the spectral curve (slopes and peaks), which is crucial for distinguishing materials.
* *Spatial Context at Multiple Scales*: Calculates the mean and standard deviation of surrounding regions of varying sizes (3x3, 5x5, 7x7, and the full 19x19 patch) to provide the model with local structural context.
* *Spatial Contrast*: Computes differences and ratios between the central pixel and the surrounding contextual windows (e.g., center vs 3x3 mean) to highlight anomalies or edges.
* *Directional Gradients*: Calculates horizontal, vertical, and diagonal differences immediately adjacent to the central pixel to capture local textures and boundaries.
* *Periphery Contrast*: Measures the difference between the central pixel and the average of the patch's four corners.
* *Spectral Band Ratios*: Segments the bands into low, mid, and high ranges, and calculates ratios and normalized differences between them (similar to common remote sensing indices like NDVI).

### Dimensionality Reduction
Due to the massive number of generated features, dimensionality reduction is applied to prevent the curse of dimensionality and improve training speed:
* *Scaling*: All extracted features are standardized using StandardScaler to have a zero mean and unit variance.
* *Principal Component Analysis (PCA)*: PCA is applied to extract the most critical variance from the data. The model retains components that explain 98% of the total variance.
* *Feature Fusion*: The final feature set is a combination (horizontal stack) of the raw scaled features and the newly generated PCA components, giving the model access to both specific localized features and global variance vectors.

### Modeling and Imbalance Handling
The chosen model is *LightGBM*, a highly efficient gradient-boosting framework.
* *Custom Class Weighting*: The training dataset suffers from class imbalance. A custom weighting algorithm calculates sample weights based on inverse class frequencies. Additionally, dominant classes (classes 3 and 4) are heavily down-weighted (multiplied by 0.05 and 0.0085 respectively) to force the model to focus on minority classes.
* *Hyperparameter Configuration*: The LightGBM classifier is tuned with a slow learning rate (0.025), a high number of estimators (900), and regularization (reg_alpha, reg_lambda) to prevent overfitting on the complex feature space. Subsampling parameters (subsample=0.75, colsample_bytree=0.65) ensure robustness by introducing randomness into the tree-building process.
