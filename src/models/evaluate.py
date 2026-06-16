"""
Model Evaluation Module
Comprehensive metrics, curves, and business impact analysis
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger("banking_ml.models.evaluate")


class ModelEvaluator:
    """
    Production model evaluation with banking-specific metrics.

    Calculates:
    - Standard ML metrics (Accuracy, Precision, Recall, F1, ROC AUC, PR AUC)
    - Business metrics (Cost analysis, ROI, Lift)
    - Visualization (ROC, PR, Confusion Matrix, Lift Chart, Gain Chart)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.metrics_history = []
        logger.info("ModelEvaluator initialized")

    def calculate_metrics(self, y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_prob: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob)
        }

        # Business metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cost_fp = self.config['business']['cost_fp']
        cost_fn = self.config['business']['cost_fn']
        avg_transaction = self.config['business']['avg_transaction_value']
        marketing_cost = self.config['business']['marketing_cost_per_customer']

        metrics['business'] = {
            'total_cost': fp * cost_fp + fn * cost_fn,
            'cost_per_customer': (fp * cost_fp + fn * cost_fn) / len(y_true),
            'revenue_opportunity': tp * avg_transaction,
            'marketing_cost': (tp + fp) * marketing_cost,
            'net_benefit': tp * avg_transaction - (fp * cost_fp + fn * cost_fn + (tp + fp) * marketing_cost),
            'roi': ((tp * avg_transaction - (fp * cost_fp + fn * cost_fn)) / 
                   ((tp + fp) * marketing_cost + 1e-10))
        }

        logger.info(f"Metrics calculated - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        return metrics

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      model_name: str = "Model", ax=None) -> None:
        """Plot ROC curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                    y_prob: np.ndarray,
                                    model_name: str = "Model", ax=None) -> None:
        """Plot Precision-Recall curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, linewidth=2, 
               label=f'{model_name} (PR AUC = {pr_auc:.4f})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             ax=None) -> None:
        """Plot confusion matrix with business context."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Transaction', 'Transaction'],
                   yticklabels=['No Transaction', 'Transaction'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix (Business Impact)', fontsize=14, fontweight='bold')

        # Add cost annotations
        cost_fp = self.config['business']['cost_fp']
        cost_fn = self.config['business']['cost_fn']
        ax.text(1.5, -0.15, f'FP Cost: ${cost_fp} each', transform=ax.transAxes, 
               fontsize=10, color='red', ha='center')
        ax.text(0.5, -0.15, f'FN Cost: ${cost_fn} each', transform=ax.transAxes,
               fontsize=10, color='red', ha='center')

    def plot_lift_chart(self, y_true: np.ndarray, y_prob: np.ndarray,
                       ax=None, n_bins: int = 10) -> None:
        """Plot Lift Chart for marketing campaign optimization."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df['decile'] = pd.qcut(df['y_prob'], n_bins, labels=False, duplicates='drop')

        decile_stats = df.groupby('decile').agg({
            'y_true': ['count', 'sum', 'mean']
        }).reset_index()
        decile_stats.columns = ['decile', 'count', 'transactions', 'response_rate']

        baseline_rate = y_true.mean()
        decile_stats['lift'] = decile_stats['response_rate'] / baseline_rate
        decile_stats['cumulative_lift'] = decile_stats['lift'].cumsum() / (decile_stats.index + 1)

        ax.bar(decile_stats['decile'], decile_stats['lift'], 
              color='steelblue', alpha=0.7, label='Decile Lift')
        ax.plot(decile_stats['decile'], decile_stats['cumulative_lift'], 
               'ro-', linewidth=2, markersize=8, label='Cumulative Lift')
        ax.axhline(y=1, color='k', linestyle='--', linewidth=1, label='Baseline')

        ax.set_xlabel('Decile (Top 10% to Bottom 10%)', fontsize=12)
        ax.set_ylabel('Lift', fontsize=12)
        ax.set_title('Lift Chart - Marketing Campaign Optimization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_gain_chart(self, y_true: np.ndarray, y_prob: np.ndarray,
                       ax=None, n_bins: int = 10) -> None:
        """Plot Cumulative Gain Chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

        n_total = len(df)
        n_positives = df['y_true'].sum()

        df['cum_population'] = (df.index + 1) / n_total * 100
        df['cum_positives'] = df['y_true'].cumsum() / n_positives * 100

        # Create deciles
        df['decile'] = pd.qcut(df.index, n_bins, labels=False, duplicates='drop')
        decile_data = df.groupby('decile').agg({
            'cum_population': 'max',
            'cum_positives': 'max'
        }).reset_index()

        ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Random')
        ax.plot(decile_data['cum_population'], decile_data['cum_positives'], 
               'b-o', linewidth=2, markersize=8, label='Model')
        ax.fill_between(decile_data['cum_population'], decile_data['cum_positives'], 
                       decile_data['cum_population'], alpha=0.2, color='blue')

        ax.set_xlabel('% of Population Targeted', fontsize=12)
        ax.set_ylabel('% of Transactions Captured', fontsize=12)
        ax.set_title('Cumulative Gain Chart', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def generate_evaluation_report(self, y_true: np.ndarray, y_prob: np.ndarray,
                                  y_pred: np.ndarray, model_name: str) -> Dict:
        """Generate comprehensive evaluation report with plots."""
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Evaluation Report: {model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)

        self.plot_roc_curve(y_true, y_prob, model_name, ax=axes[0, 0])
        self.plot_precision_recall_curve(y_true, y_prob, model_name, ax=axes[0, 1])
        self.plot_confusion_matrix(y_true, y_pred, ax=axes[0, 2])
        self.plot_lift_chart(y_true, y_prob, ax=axes[1, 0])
        self.plot_gain_chart(y_true, y_prob, ax=axes[1, 1])

        # Business impact summary
        ax = axes[1, 2]
        ax.axis('off')
        business = metrics['business']
        summary_text = f"""
        BUSINESS IMPACT SUMMARY
        ─────────────────────────

        Total Cost: ${business['total_cost']:,.0f}
        Cost per Customer: ${business['cost_per_customer']:.2f}

        Revenue Opportunity: ${business['revenue_opportunity']:,.0f}
        Marketing Cost: ${business['marketing_cost']:,.0f}

        Net Benefit: ${business['net_benefit']:,.0f}
        ROI: {business['roi']:.2f}x

        ─────────────────────────
        ML METRICS
        Accuracy:  {metrics['accuracy']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall:    {metrics['recall']:.4f}
        F1 Score:  {metrics['f1']:.4f}
        ROC AUC:   {metrics['roc_auc']:.4f}
        PR AUC:    {metrics['pr_auc']:.4f}
        """
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        report = {
            'model_name': model_name,
            'metrics': metrics,
            'figure': fig,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        self.metrics_history.append(report)
        logger.info(f"Evaluation report generated for {model_name}")
        return report

    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models and create ranking."""
        comparison = []
        for name, result in results.items():
            if result.get('cv_mean_auc') is not None:
                comparison.append({
                    'Model': name,
                    'CV_AUC_Mean': result['cv_mean_auc'],
                    'CV_AUC_Std': result['cv_std_auc'],
                    'Status': 'Success'
                })
            else:
                comparison.append({
                    'Model': name,
                    'CV_AUC_Mean': None,
                    'CV_AUC_Std': None,
                    'Status': 'Failed'
                })

        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('CV_AUC_Mean', ascending=False)

        logger.info("Model comparison completed")
        return df_comparison
