from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

class AnomalyDetector:
    """
    Dual-path anomaly detector:
    1. Deep feature path: Isolation Forest on PCA-reduced deep features
    2. Forensic feature path: Mahalanobis distance on forensic features (ELA, noise)
    
    The two paths are combined with learned weights for final scoring.
    Training calibration establishes data-driven thresholds.
    """
    def __init__(self, contamination='auto', n_estimators=300, random_state=42, use_pca=True):
        self.use_pca = use_pca
        self.pca = None
        self.random_state = random_state
        
        # Deep feature detector
        self.deep_model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_samples='auto',
            bootstrap=True,
            n_jobs=-1
        )
        
        # Forensic feature detector
        self.forensic_scaler = StandardScaler()
        self.forensic_covariance = None
        
        # Calibration
        self.calibration = {}
        self.is_fitted = False

    def train(self, deep_features, forensic_features):
        """
        Train both detection paths.
        deep_features: (N_patches, deep_dim) — from FeatureExtractor
        forensic_features: (N_images, forensic_dim) — from ForensicFeatureExtractor
        """
        deep_features = np.nan_to_num(deep_features)
        forensic_features = np.nan_to_num(forensic_features)
        
        print(f"Training with {len(deep_features)} deep patches, "
              f"{len(forensic_features)} forensic image vectors...")
        
        # === Deep Feature Path ===
        X_deep = deep_features
        if self.use_pca:
            n_comp = min(0.95, deep_features.shape[1], deep_features.shape[0])
            self.pca = PCA(n_components=0.95, random_state=self.random_state, svd_solver='full')
            X_deep = self.pca.fit_transform(deep_features)
            print(f"Deep PCA: {deep_features.shape[1]} → {X_deep.shape[1]} dims")
        
        self.deep_model.fit(X_deep)
        
        # Deep scores on training data
        deep_scores = -self.deep_model.decision_function(X_deep)
        self.calibration['deep_mean'] = float(np.mean(deep_scores))
        self.calibration['deep_std'] = float(np.std(deep_scores)) + 1e-8
        
        # === Forensic Feature Path ===
        X_forensic = self.forensic_scaler.fit_transform(forensic_features)
        
        try:
            if X_forensic.shape[0] > X_forensic.shape[1] * 2:
                self.forensic_covariance = MinCovDet(random_state=self.random_state)
            else:
                self.forensic_covariance = EmpiricalCovariance()
            self.forensic_covariance.fit(X_forensic)
        except Exception as e:
            print(f"Warning: Robust covariance failed ({e}), using empirical.")
            self.forensic_covariance = EmpiricalCovariance()
            self.forensic_covariance.fit(X_forensic)
        
        forensic_scores = self.forensic_covariance.mahalanobis(X_forensic)
        self.calibration['forensic_mean'] = float(np.mean(forensic_scores))
        self.calibration['forensic_std'] = float(np.std(forensic_scores)) + 1e-8
        
        print(f"Deep scores: mean={self.calibration['deep_mean']:.4f}, "
              f"std={self.calibration['deep_std']:.4f}")
        print(f"Forensic scores: mean={self.calibration['forensic_mean']:.4f}, "
              f"std={self.calibration['forensic_std']:.4f}")
        
        self.is_fitted = True

    def predict_deep(self, deep_features):
        """Score deep features using Isolation Forest."""
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")
        
        X = deep_features
        if self.use_pca:
            X = self.pca.transform(deep_features)
        
        raw_scores = -self.deep_model.decision_function(X)
        normalized = (raw_scores - self.calibration['deep_mean']) / self.calibration['deep_std']
        return normalized

    def predict_forensic(self, forensic_features):
        """Score forensic features using Mahalanobis distance."""
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")
        
        X = self.forensic_scaler.transform(forensic_features.reshape(1, -1) 
                                            if forensic_features.ndim == 1 
                                            else forensic_features)
        raw_scores = self.forensic_covariance.mahalanobis(X)
        normalized = (raw_scores - self.calibration['forensic_mean']) / self.calibration['forensic_std']
        return normalized

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'deep_model': self.deep_model,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'forensic_scaler': self.forensic_scaler,
            'forensic_covariance': self.forensic_covariance,
            'calibration': self.calibration,
        }
        joblib.dump(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        state = joblib.load(path)
        self.deep_model = state['deep_model']
        self.pca = state.get('pca', None)
        self.use_pca = state.get('use_pca', False)
        self.forensic_scaler = state.get('forensic_scaler', StandardScaler())
        self.forensic_covariance = state.get('forensic_covariance', None)
        self.calibration = state.get('calibration', {})
        self.is_fitted = True
        print(f"Model loaded from {path}")
