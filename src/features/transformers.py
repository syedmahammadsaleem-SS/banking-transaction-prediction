"""
Feature Transformation Module
Scaling, encoding, and transformation operations
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from typing import Optional
import yaml

logger = logging.getLogger("banking_ml.features.transformers")


class FeatureTransformer:
    """
    Production feature transformation pipeline.

    Handles:
    - Feature scaling (Standard, Robust, MinMax)
    - Power transformation (Yeo-Johnson)
    - Skewness correction
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scalers = {}
        self.transformers = {}
        logger.info("FeatureTransformer initialized")

    def fit_scaler(self, df: pd.DataFrame, scaler_type: str = "robust") -> None:
        """Fit scaler on training data."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        scaler.fit(df[feature_cols])
        self.scalers[scaler_type] = scaler
        self.feature_cols = feature_cols
        logger.info(f"Fitted {scaler_type} scaler on {len(feature_cols)} features")

    def transform(self, df: pd.DataFrame, scaler_type: str = "robust") -> pd.DataFrame:
        """Apply fitted scaler."""
        df_scaled = df.copy()
        scaler = self.scalers[scaler_type]
        df_scaled[self.feature_cols] = scaler.transform(df[self.feature_cols])
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, scaler_type: str = "robust") -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit_scaler(df, scaler_type)
        return self.transform(df, scaler_type)

    def correct_skewness(self, df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
        """
        Apply Yeo-Johnson power transformation to correct skewness.

        Args:
            df: Input dataframe
            threshold: Skewness threshold for transformation
        """
        df_transformed = df.copy()
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        skewed_features = []
        for col in feature_cols:
            skewness = df[col].skew()
            if abs(skewness) > threshold:
                skewed_features.append(col)

        if skewed_features:
            pt = PowerTransformer(method='yeo-johnson')
            df_transformed[skewed_features] = pt.fit_transform(df[skewed_features])
            self.transformers['power'] = pt
            self.transformed_features = skewed_features
            logger.info(f"Corrected skewness for {len(skewed_features)} features")

        return df_transformed
