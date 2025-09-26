# preprocess.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional

def normalize_pixels(X: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0,255] to [0,1] and return float32 array."""
    return X.astype(np.float32) / 255.0

def fit_scaler(X: np.ndarray):
    """Fit StandardScaler on X and return (scaler, X_scaled)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return scaler, Xs

def transform_with_scaler(scaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)

def fit_pca(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, n_components: int = 100):
    pca = PCA(n_components=n_components)
    Xtr = pca.fit_transform(X_train_scaled)
    Xte = pca.transform(X_test_scaled)
    return pca, Xtr, Xte

# utility to process full pipeline (used inside train.py or evaluate.py)
def prepare_data(X_train, X_test, y_train=None, subset: int = 0, do_pca: Optional[int]=None):
    X_train = normalize_pixels(X_train)
    X_test  = normalize_pixels(X_test)
    if subset and subset > 0:
        X_train = X_train[:subset]
        if y_train is not None:
            y_train = y_train[:subset]
    scaler, X_train_scaled = fit_scaler(X_train)
    X_test_scaled = transform_with_scaler(scaler, X_test)
    pca = None
    if do_pca and do_pca > 0:
        pca, X_train_scaled, X_test_scaled = fit_pca(X_train_scaled, X_test_scaled, n_components=do_pca)
    if y_train is not None:
        return scaler, pca, X_train_scaled, X_test_scaled, y_train
    return scaler, pca, X_train_scaled, X_test_scaled
