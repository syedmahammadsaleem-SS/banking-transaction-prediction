"""
Feature Engineering Builders
Advanced feature creation for banking transaction prediction
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import PolynomialFeatures
import yaml

logger = logging.getLogger("banking_ml.features.builders")


class FeatureBuilder:
    """
    Production-grade feature engineering for banking data.

    Creates:
    - Statistical aggregations (mean, std, min, max, skew, kurtosis)
    - Feature interactions (ratios, differences)
    - Polynomial features
    - Domain-specific banking features
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.id_col = self.config['data']['id_column']
        self.target_col = self.config['data']['target_column']
        logger.info("FeatureBuilder initialized")

    def build_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregation features across all numeric columns.

        Features:
        - row_mean: Mean of all features per row
        - row_std: Standard deviation per row
        - row_min: Minimum value per row
        - row_max: Maximum value per row
        - row_skew: Skewness per row
        - row_kurt: Kurtosis per row
        - row_median: Median per row
        - row_sum: Sum per row
        - row_range: Max - Min per row
        - row_iqr: Interquartile range per row
        """
        df_new = df.copy()
        feature_cols = [c for c in df.columns if c not in [self.id_col, self.target_col]]

        # Basic statistics
        df_new['row_mean'] = df[feature_cols].mean(axis=1)
        df_new['row_std'] = df[feature_cols].std(axis=1)
        df_new['row_min'] = df[feature_cols].min(axis=1)
        df_new['row_max'] = df[feature_cols].max(axis=1)
        df_new['row_median'] = df[feature_cols].median(axis=1)
        df_new['row_sum'] = df[feature_cols].sum(axis=1)
        df_new['row_range'] = df_new['row_max'] - df_new['row_min']
        df_new['row_iqr'] = df[feature_cols].quantile(0.75, axis=1) - df[feature_cols].quantile(0.25, axis=1)

        # Advanced statistics
        df_new['row_skew'] = df[feature_cols].skew(axis=1)
        df_new['row_kurt'] = df[feature_cols].kurtosis(axis=1)
        df_new['row_var'] = df[feature_cols].var(axis=1)
        df_new['row_mad'] = df[feature_cols].mad(axis=1)

        # Percentile features
        df_new['row_p10'] = df[feature_cols].quantile(0.10, axis=1)
        df_new['row_p90'] = df[feature_cols].quantile(0.90, axis=1)
        df_new['row_p95'] = df[feature_cols].quantile(0.95, axis=1)
        df_new['row_p05'] = df[feature_cols].quantile(0.05, axis=1)

        # Zero and negative counts
        df_new['row_n_zeros'] = (df[feature_cols] == 0).sum(axis=1)
        df_new['row_n_negatives'] = (df[feature_cols] < 0).sum(axis=1)
        df_new['row_n_positives'] = (df[feature_cols] > 0).sum(axis=1)

        logger.info(f"Built {len(df_new.columns) - len(df.columns)} statistical features")
        return df_new

    def build_interaction_features(self, df: pd.DataFrame, 
                                   top_features: Optional[List[str]] = None,
                                   n_interactions: int = 50) -> pd.DataFrame:
        """
        Create interaction features (ratios and differences).

        Args:
            df: Input dataframe
            top_features: List of features to create interactions for
            n_interactions: Number of top interactions to create
        """
        df_new = df.copy()

        if top_features is None:
            feature_cols = [c for c in df.columns if c not in [self.id_col, self.target_col]]
            top_features = feature_cols[:n_interactions]

        # Create ratio and difference features for top pairs
        for i in range(len(top_features)):
            for j in range(i+1, min(i+5, len(top_features))):  # Limit combinations
                f1, f2 = top_features[i], top_features[j]

                # Ratio (with safety for division by zero)
                df_new[f'{f1}_div_{f2}'] = np.where(
                    df[f2] != 0, 
                    df[f1] / df[f2], 
                    0
                )

                # Difference
                df_new[f'{f1}_minus_{f2}'] = df[f1] - df[f2]

                # Product
                df_new[f'{f1}_times_{f2}'] = df[f1] * df[f2]

        logger.info(f"Built interaction features for {len(top_features)} base features")
        return df_new

    def build_polynomial_features(self, df: pd.DataFrame, 
                                  degree: int = 2,
                                  interaction_only: bool = False) -> pd.DataFrame:
        """
        Create polynomial features for top predictive features.

        Args:
            df: Input dataframe
            degree: Polynomial degree
            interaction_only: If True, only interaction terms
        """
        df_new = df.copy()
        feature_cols = [c for c in df.columns if c not in [self.id_col, self.target_col]]

        # Select top features by variance to avoid explosion
        top_features = df[feature_cols].var().nlargest(20).index.tolist()

        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(df[top_features])

        feature_names = poly.get_feature_names_out(top_features)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        # Remove original features to avoid duplication
        poly_df = poly_df.drop(columns=top_features)

        df_new = pd.concat([df_new, poly_df], axis=1)
        logger.info(f"Built polynomial features: {poly_df.shape[1]} new features")
        return df_new

    def build_banking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific banking features.

        Features:
        - Volatility indicators
        - Trend indicators
        - Concentration measures
        - Risk indicators
        """
        df_new = df.copy()
        feature_cols = [c for c in df.columns if c not in [self.id_col, self.target_col]]

        # Volatility (coefficient of variation)
        df_new['cv'] = df_new['row_std'] / (df_new['row_mean'] + 1e-10)

        # Concentration (Gini-like coefficient)
        def gini_coefficient(x):
            x = np.array(x)
            x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(x)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

        df_new['gini'] = df[feature_cols].apply(gini_coefficient, axis=1)

        # Trend indicators (first vs second half)
        mid = len(feature_cols) // 2
        first_half = feature_cols[:mid]
        second_half = feature_cols[mid:]
        df_new['trend_first_half'] = df[first_half].mean(axis=1)
        df_new['trend_second_half'] = df[second_half].mean(axis=1)
        df_new['trend_diff'] = df_new['trend_second_half'] - df_new['trend_first_half']

        # Risk indicators
        df_new['extreme_count'] = ((df[feature_cols] > df[feature_cols].quantile(0.99)) | 
                                   (df[feature_cols] < df[feature_cols].quantile(0.01))).sum(axis=1)

        # Stability (how many features are within 1 std of mean)
        df_new['stability'] = ((df[feature_cols] - df_new['row_mean'].values.reshape(-1,1)).abs() < 
                               df_new['row_std'].values.reshape(-1,1)).sum(axis=1) / len(feature_cols)

        logger.info(f"Built {len(df_new.columns) - len(df.columns)} banking domain features")
        return df_new

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")

        df = self.build_statistical_features(df)
        df = self.build_interaction_features(df)
        df = self.build_banking_features(df)
        # Polynomial features optional due to dimensionality
        # df = self.build_polynomial_features(df)

        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
