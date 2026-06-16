"""
Hyperparameter Tuning Module
Random Search, Grid Search, and Optuna optimization
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import yaml

logger = logging.getLogger("banking_ml.models.tune")

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    Production hyperparameter tuning with multiple strategies.

    Supports:
    - Random Search
    - Grid Search
    - Optuna (Bayesian Optimization)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.best_params = {}
        self.study = None
        logger.info("HyperparameterTuner initialized")

    def get_param_distributions(self, model_name: str) -> Dict:
        """Get parameter distributions for Random Search."""
        distributions = {
            'RandomForest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [20, 31, 50, 100],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_samples': [10, 20, 30],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        return distributions.get(model_name, {})

    def random_search(self, X: pd.DataFrame, y: pd.Series,
                     model_name: str,
                     n_iter: int = 50,
                     cv_folds: int = 5) -> Dict:
        """
        Perform Randomized Search CV.

        Args:
            X: Features
            y: Target
            model_name: Model to tune
            n_iter: Number of parameter settings sampled
            cv_folds: Number of CV folds
        """
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        import lightgbm as lgb

        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        }

        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported for tuning")

        model = models[model_name]
        param_dist = self.get_param_distributions(model_name)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter,
            scoring='roc_auc', cv=cv, n_jobs=-1,
            random_state=42, verbose=1
        )

        search.fit(X, y)

        self.best_params[model_name] = search.best_params_

        logger.info(f"Random Search complete for {model_name}")
        logger.info(f"Best CV AUC: {search.best_score_:.4f}")
        logger.info(f"Best params: {search.best_params_}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'best_model': search.best_estimator_
        }

    def grid_search(self, X: pd.DataFrame, y: pd.Series,
                   model_name: str,
                   param_grid: Optional[Dict] = None,
                   cv_folds: int = 5) -> Dict:
        """Perform exhaustive Grid Search CV."""
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        }

        model = models[model_name]

        if param_grid is None:
            param_grid = self.get_param_distributions(model_name)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        search = GridSearchCV(
            model, param_grid,
            scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1
        )

        search.fit(X, y)

        self.best_params[model_name] = search.best_params_

        logger.info(f"Grid Search complete for {model_name}")
        logger.info(f"Best CV AUC: {search.best_score_:.4f}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'best_model': search.best_estimator_
        }

    def optuna_optimize(self, X: pd.DataFrame, y: pd.Series,
                       model_name: str,
                       n_trials: int = 100,
                       cv_folds: int = 5) -> Dict:
        """
        Bayesian optimization with Optuna.

        Most efficient for high-dimensional search spaces.
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        def objective(trial):
            if model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)

            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                    'random_state': 42,
                    'n_jobs': -1,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)

            elif model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                    'random_seed': 42,
                    'verbose': False
                }
                model = cb.CatBoostClassifier(**params)

            else:
                raise ValueError(f"Optuna not configured for {model_name}")

            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, y_prob))

            return np.mean(scores)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.study = study
        self.best_params[model_name] = study.best_params

        logger.info(f"Optuna optimization complete for {model_name}")
        logger.info(f"Best CV AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
            'n_trials': n_trials
        }
