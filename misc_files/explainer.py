# -*- coding: utf-8 -*-
"""
DTRA Explainer Module
SHAP-based interpretability for threat detection decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import SHAP (optional - graceful fallback if not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")


class ThreatExplainer:
    """
    SHAP-based explainer for DNN predictions.
    Provides both global feature importance and individual alert explanations.
    """
    
    def __init__(self, detector=None):
        self.detector = detector
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        
    def setup(self, background_samples, feature_names=None):
        """
        Initialize the SHAP explainer with background data.
        
        Args:
            background_samples: DataFrame or array of typical samples (100-500 rows)
            feature_names: list of feature column names
        """
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not installed. Run: pip install shap")
        
        if self.detector is None or self.detector.model is None:
            raise RuntimeError("Detector model not loaded. Call detector.load() first.")
        
        # Store feature names
        if isinstance(background_samples, pd.DataFrame):
            self.feature_names = list(background_samples.columns)
            background_array = background_samples.values
        else:
            self.feature_names = feature_names or [f"Feature_{i}" for i in range(background_samples.shape[1])]
            background_array = background_samples
        
        # Preprocess background data
        bg_imputed = self.detector.imputer.transform(background_array)
        bg_scaled = self.detector.scaler.transform(bg_imputed)
        
        self.background_data = bg_scaled
        
        # Create DeepExplainer
        print("🔍 Initializing SHAP DeepExplainer...")
        self.explainer = shap.DeepExplainer(self.detector.model, self.background_data)
        print("✅ Explainer ready")
        
    def explain_batch(self, X, num_samples=100):
        """
        Compute SHAP values for a batch of samples.
        
        Args:
            X: DataFrame or array of samples to explain
            num_samples: max samples to process
            
        Returns:
            numpy array of SHAP values
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call setup() first.")
        
        # Preprocess
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X[:num_samples]
        X_imputed = self.detector.imputer.transform(X)
        X_scaled = self.detector.scaler.transform(X_imputed)
        
        # Compute SHAP values
        print(f"🧮 Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        while len(shap_values.shape) > 2:
            shap_values = shap_values.squeeze(axis=-1)
        
        print("✅ SHAP computation complete")
        return shap_values
    
    def plot_global_importance(self, X, shap_values=None, max_display=20):
        """
        Generate global feature importance summary plot.
        
        Args:
            X: DataFrame of samples (for feature names and values)
            shap_values: pre-computed SHAP values (optional)
            max_display: number of features to show
        """
        if shap_values is None:
            shap_values = self.explain_batch(X)
        
        # Prepare display data
        if isinstance(X, pd.DataFrame):
            display_data = X.head(len(shap_values))
        else:
            display_data = pd.DataFrame(X[:len(shap_values)], columns=self.feature_names)
        
        print("\n📊 Global Feature Importance")
        print("=" * 50)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            display_data.values,
            feature_names=list(display_data.columns),
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Interpretation:")
        print("• Top features have strongest influence on predictions")
        print("• Red = High values push toward 'Attack'")
        print("• Blue = Low values push toward 'Benign'")
    
    def explain_single_alert(self, sample_index, X, shap_values=None, top_k=10):
        """
        Explain a single alert with detailed feature breakdown.
        
        Args:
            sample_index: index of sample to explain
            X: DataFrame of all samples
            shap_values: pre-computed SHAP values
            top_k: number of top features to show
            
        Returns:
            DataFrame of feature contributions
        """
        if shap_values is None:
            shap_values = self.explain_batch(X)
        
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = self.feature_names
        
        # Get contributions for this sample
        contributions = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[sample_index]
        })
        contributions['Abs Impact'] = contributions['SHAP Value'].abs()
        contributions = contributions.sort_values('Abs Impact', ascending=False).head(top_k)
        
        print(f"\n🎯 Alert #{sample_index} Explanation")
        print("=" * 50)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        colors = ['#FF4444' if x > 0 else '#4444FF' for x in contributions['SHAP Value']]
        plt.barh(contributions['Feature'], contributions['SHAP Value'], color=colors)
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title(f'Top {top_k} Influential Features for Alert #{sample_index}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.show()
        
        print("\n💡 Legend:")
        print("• 🔴 Red bars = Pushing toward ATTACK")
        print("• 🔵 Blue bars = Pushing toward BENIGN")
        
        return contributions
    
    def get_top_factors(self, sample_index, X, shap_values=None, top_k=3):
        """
        Get top contributing factors for an alert (for dashboard display).
        
        Returns:
            list of dicts with feature info
        """
        if shap_values is None:
            shap_values = self.explain_batch(X)
        
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            values = X.iloc[sample_index].values
        else:
            feature_names = self.feature_names
            values = X[sample_index]
        
        abs_impacts = np.abs(shap_values[sample_index])
        top_indices = np.argsort(abs_impacts)[::-1][:top_k]
        
        factors = []
        for idx in top_indices:
            factors.append({
                'feature': feature_names[idx],
                'value': values[idx],
                'impact': shap_values[sample_index][idx],
                'direction': 'Attack' if shap_values[sample_index][idx] > 0 else 'Benign'
            })
        
        return factors
