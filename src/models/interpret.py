"""
Model Interpretability Module
SHAP, Feature Importance, Permutation Importance
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger("banking_ml.models.interpret")


class ModelInterpreter:
    """
    Production model interpretability with SHAP and permutation importance.

    Provides:
    - Global feature importance
    - Local explanations for individual predictions
    - Summary plots
    - Waterfall plots for business stakeholders
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.explainer = None
        self.shap_values = None
        logger.info("ModelInterpreter initialized")

    def explain_with_shap(self, model: object, X: pd.DataFrame,
                         sample_size: int = 1000) -> Dict:
        """
        Generate SHAP explanations for model predictions.

        Args:
            model: Trained model
            X: Feature dataframe
            sample_size: Number of samples for SHAP computation
        """
        X_sample = X.sample(min(sample_size, len(X)), random_state=42)

        try:
            # Tree-based models
            if hasattr(model, 'tree_'):
                self.explainer = shap.TreeExplainer(model)
            else:
                self.explainer = shap.KernelExplainer(model.predict_proba, X_sample)

            self.shap_values = self.explainer.shap_values(X_sample)

            # Handle binary classification
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[1]  # Class 1 (Transaction)
            else:
                shap_vals = self.shap_values

            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_vals).mean(axis=0)
            feature_importance = dict(zip(X.columns, mean_shap))

            # Top features
            top_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:20]

            logger.info(f"SHAP explanation generated for {len(X_sample)} samples")

            return {
                'explainer': self.explainer,
                'shap_values': shap_vals,
                'feature_importance': feature_importance,
                'top_features': top_features,
                'sample_data': X_sample
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return {'error': str(e)}

    def plot_shap_summary(self, shap_values: np.ndarray, 
                         X_sample: pd.DataFrame,
                         max_display: int = 20) -> plt.Figure:
        """Plot SHAP summary plot (beeswarm)."""
        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, 
                         show=False, plot_size=None)
        plt.title('SHAP Feature Importance - Global Explanation', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def plot_shap_waterfall(self, shap_values: np.ndarray,
                           X_sample: pd.DataFrame,
                           instance_idx: int = 0) -> plt.Figure:
        """Plot SHAP waterfall for single prediction explanation."""
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[instance_idx],
            base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            data=X_sample.iloc[instance_idx],
            feature_names=X_sample.columns
        ), show=False)
        plt.title(f'SHAP Waterfall - Customer {instance_idx} Prediction Explanation',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def calculate_permutation_importance(self, model: object,
                                       X: pd.DataFrame, y: pd.Series,
                                       n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance for model validation.

        More reliable than built-in feature importance as it breaks
        feature-target relationship.
        """
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats,
            random_state=42, scoring='roc_auc', n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        logger.info(f"Permutation importance calculated for {len(X.columns)} features")
        return importance_df

    def plot_permutation_importance(self, importance_df: pd.DataFrame,
                                   top_n: int = 20) -> plt.Figure:
        """Plot permutation importance with error bars."""
        fig, ax = plt.subplots(figsize=(10, 12))

        top = importance_df.head(top_n)
        colors = ['#2E86AB' if imp > 0 else '#A23B72' 
                 for imp in top['importance_mean']]

        ax.barh(range(len(top)), top['importance_mean'], 
               xerr=top['importance_std'], color=colors, alpha=0.8, capsize=3)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance (ROC AUC decrease)', fontsize=12)
        ax.set_title('Permutation Feature Importance
(More Reliable than Built-in Importance)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linewidth=0.8)

        plt.tight_layout()
        return fig

    def explain_prediction(self, model: object, X: pd.DataFrame,
                          instance: pd.Series or int,
                          feature_names: Optional[List] = None) -> Dict:
        """
        Explain a single prediction for business stakeholders.

        Returns human-readable explanation of why model predicted transaction/no-transaction.
        """
        if isinstance(instance, int):
            instance_data = X.iloc[instance:instance+1]
        else:
            instance_data = instance.values.reshape(1, -1)

        prediction_prob = model.predict_proba(instance_data)[0, 1]
        prediction = 1 if prediction_prob > 0.5 else 0

        # Get SHAP values for this instance
        if self.explainer is not None:
            shap_vals = self.explainer.shap_values(instance_data)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1][0]
            else:
                shap_vals = shap_vals[0]

            # Top contributing features
            feature_contrib = list(zip(X.columns, shap_vals))
            feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

            top_positive = [(f, v) for f, v in feature_contrib if v > 0][:5]
            top_negative = [(f, v) for f, v in feature_contrib if v < 0][:5]

            explanation = {
                'prediction': 'Transaction' if prediction == 1 else 'No Transaction',
                'probability': prediction_prob,
                'confidence': max(prediction_prob, 1 - prediction_prob),
                'top_drivers_transaction': top_positive,
                'top_drivers_no_transaction': top_negative,
                'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
            }
        else:
            explanation = {
                'prediction': 'Transaction' if prediction == 1 else 'No Transaction',
                'probability': prediction_prob,
                'confidence': max(prediction_prob, 1 - prediction_prob),
                'note': 'SHAP explainer not initialized. Run explain_with_shap first.'
            }

        return explanation

    def generate_interpretability_report(self, model: object, 
                                        X: pd.DataFrame, y: pd.Series) -> Dict:
        """Generate comprehensive interpretability report."""
        # SHAP explanations
        shap_results = self.explain_with_shap(model, X)

        # Permutation importance
        perm_importance = self.calculate_permutation_importance(model, X, y)

        report = {
            'shap': shap_results,
            'permutation_importance': perm_importance,
            'top_shap_features': shap_results.get('top_features', []),
            'top_permutation_features': perm_importance.head(20).to_dict('records')
        }

        logger.info("Interpretability report generated")
        return report
