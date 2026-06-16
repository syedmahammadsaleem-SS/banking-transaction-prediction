"""
Model Training Module
Train and compare 12+ ML models with cross-validation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import yaml
import pickle
import os
from datetime import datetime

logger = logging.getLogger("banking_ml.models.train")


class ModelTrainer:
    """
    Production model training pipeline.

    Trains 12 models:
    1. Logistic Regression
    2. Naive Bayes
    3. Decision Tree
    4. Random Forest
    5. Extra Trees
    6. AdaBoost
    7. Gradient Boosting
    8. XGBoost
    9. LightGBM
    10. CatBoost
    11. KNN
    12. SVM
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.models = {}
        self.cv_results = {}
        self.trained_models = {}
        logger.info("ModelTrainer initialized")

    def get_base_models(self) -> Dict[str, object]:
        """Initialize all base models with default parameters."""
        models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(
                random_state=42, max_depth=10, class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                eval_metric='logloss', use_label_encoder=False
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                verbose=-1
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=100, random_seed=42, verbose=False,
                loss_function='Logloss'
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
        }
        return models

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series,
                     models: Optional[Dict] = None,
                     cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train models with stratified k-fold cross-validation.

        Returns:
            Dictionary with model names and CV scores
        """
        if models is None:
            models = self.get_base_models()

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = {}

        for name, model in models.items():
            logger.info(f"Training {name} with {cv_folds}-fold CV...")

            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

                # Fit on full data for feature importance
                model.fit(X, y)

                results[name] = {
                    'cv_mean_auc': cv_scores.mean(),
                    'cv_std_auc': cv_scores.std(),
                    'cv_scores': cv_scores,
                    'model': model,
                    'fitted': True
                }

                logger.info(f"{name}: CV AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                results[name] = {
                    'cv_mean_auc': None,
                    'cv_std_auc': None,
                    'cv_scores': None,
                    'model': None,
                    'fitted': False,
                    'error': str(e)
                }

        self.cv_results = results
        self.trained_models = {k: v['model'] for k, v in results.items() if v['fitted']}
        return results

    def train_single_model(self, X: pd.DataFrame, y: pd.Series,
                          model_name: str,
                          model_params: Optional[Dict] = None) -> object:
        """Train a single model with custom parameters."""
        models = self.get_base_models()

        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")

        model = models[model_name]
        if model_params:
            model.set_params(**model_params)

        model.fit(X, y)
        logger.info(f"Trained {model_name} on full dataset")
        return model

    def save_model(self, model: object, model_name: str,
                  filepath: Optional[str] = None) -> str:
        """Save trained model with versioning."""
        if filepath is None:
            model_dir = self.config['paths']['models']
            os.makedirs(model_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{model_dir}/{model_name}_{timestamp}.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Saved model to {filepath}")
        return filepath

    def load_model(self, filepath: str) -> object:
        """Load saved model."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {filepath}")
        return model
