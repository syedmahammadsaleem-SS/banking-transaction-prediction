"""
Data Transformation Module
Handles cleaning, normalization, and preprocessing
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import yaml

logger = logging.getLogger("banking_ml.data.transform")


class DataTransformer:
    """
    Production-grade data transformation pipeline.

    Implements:
    - Missing value handling
    - Outlier treatment
    - Feature scaling
    - Data type optimization
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = None
        logger.info("DataTransformer initialized")

    def handle_missing(self, df: pd.DataFrame, 
                       strategy: str = "median") -> pd.DataFrame:
        """
        Handle missing values with configurable strategy.

        Strategies: median, mean, mode, drop, interpolate
        """
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()

        if missing_before == 0:
            logger.info("No missing values found")
            return df_clean

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                if strategy == "median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif strategy == "mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif strategy == "interpolate":
                    df_clean[col].interpolate(method='linear', inplace=True)

        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        return df_clean

    def treat_outliers(self, df: pd.DataFrame, 
                       method: str = "iqr",
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Treat outliers using IQR or Z-score method.

        Args:
            df: Input dataframe
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        outlier_counts = {}
        for col in numeric_cols:
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
                outlier_counts[col] = outliers
                df_clean[col] = df_clean[col].clip(lower, upper)
            elif method == "zscore":
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = (z_scores > threshold).sum()
                outlier_counts[col] = outliers
                df_clean[col] = np.where(z_scores > threshold, 
                                        df_clean[col].median(), 
                                        df_clean[col])

        total_outliers = sum(outlier_counts.values())
        logger.info(f"Outliers treated: {total_outliers} values across {len(numeric_cols)} columns")
        return df_clean

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by downcasting numeric types."""
        df_opt = df.copy()

        for col in df_opt.select_dtypes(include=['int']).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')

        for col in df_opt.select_dtypes(include=['float']).columns:
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')

        mem_before = df.memory_usage(deep=True).sum() / 1024**2
        mem_after = df_opt.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory optimized: {mem_before:.2f}MB -> {mem_after:.2f}MB")
        return df_opt

    def fit_scaler(self, df: pd.DataFrame, scaler_type: str = "robust"):
        """Fit scaler on training data."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['target_column'], 
                        self.config['data']['id_column']]]

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()

        self.scaler.fit(df[feature_cols])
        self.feature_cols = feature_cols
        logger.info(f"Fitted {scaler_type} scaler on {len(feature_cols)} features")

    def transform_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to data."""
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled
