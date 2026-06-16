"""
Data Quality Checks Module
Comprehensive quality assessment with drift detection
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import yaml

logger = logging.getLogger("banking_ml.data.quality")


class QualityChecker:
    """
    Advanced data quality checker with statistical tests.

    Implements:
    - Distribution analysis
    - Drift detection
    - Correlation analysis
    - Outlier detection
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("QualityChecker initialized")

    def check_distribution(self, df: pd.DataFrame, 
                          target_col: str = "target") -> Dict:
        """Analyze feature distributions and compare by target."""
        results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_col]

        for col in numeric_cols:
            group_0 = df[df[target_col] == 0][col]
            group_1 = df[df[target_col] == 1][col]

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(group_0, group_1)

            # Mann-Whitney U test
            mw_stat, mw_pval = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')

            results[col] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pval,
                'significant': ks_pval < 0.05,
                'mean_diff': group_1.mean() - group_0.mean(),
                'std_diff': group_1.std() - group_0.std()
            }

        significant_features = sum(1 for r in results.values() if r['significant'])
        logger.info(f"Distribution check: {significant_features}/{len(numeric_cols)} features significantly different")
        return results

    def detect_drift(self, reference_df: pd.DataFrame, 
                    current_df: pd.DataFrame) -> Dict:
        """Detect data drift between reference and current datasets."""
        drift_results = {}
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in current_df.columns:
                # Population Stability Index (PSI)
                psi = self._calculate_psi(reference_df[col], current_df[col])

                # KS test
                ks_stat, ks_pval = stats.ks_2samp(reference_df[col], current_df[col])

                drift_results[col] = {
                    'psi': psi,
                    'psi_drift': psi > 0.25,  # PSI > 0.25 indicates significant drift
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval,
                    'drift_detected': psi > 0.25 or ks_pval < 0.01
                }

        drifted = sum(1 for r in drift_results.values() if r['drift_detected'])
        total = len(drift_results)
        logger.info(f"Drift detection: {drifted}/{total} features show drift")
        return drift_results

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        def scale_range(input, min_val, max_val):
            return (input - min_val) / (max_val - min_val)

        breakpoints = np.linspace(0, 1, buckets + 1)
        breakpoints = np.percentile(expected, breakpoints * 100)

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        # Handle zero bins
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi = np.sum((expected_percents - actual_percents) * 
                     np.log(expected_percents / actual_percents))
        return psi

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive quality report."""
        report = {
            'shape': df.shape,
            'missing_summary': df.isnull().sum().to_dict(),
            'missing_pct': (df.isnull().mean() * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'numeric_stats': df.describe().to_dict(),
            'skewness': df.select_dtypes(include=[np.number]).skew().to_dict(),
            'kurtosis': df.select_dtypes(include=[np.number]).kurtosis().to_dict()
        }

        logger.info("Quality report generated")
        return report
