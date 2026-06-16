"""
Feature Selection Module
Multiple selection strategies with ranking report
"""

import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger("banking_ml.features.selectors")


class FeatureSelector:
    """
    Production feature selection with multiple strategies.

    Methods:
    - Variance Threshold
    - Mutual Information
    - SelectKBest
    - RFE (Recursive Feature Elimination)
    - Feature Importance (Random Forest)
    - SHAP Importance
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.selected_features = None
        self.feature_rankings = {}
        logger.info("FeatureSelector initialized")

    def variance_threshold(self, df: pd.DataFrame, 
                         threshold: Optional[float] = None) -> List[str]:
        """Remove low-variance features."""
        if threshold is None:
            threshold = self.config['data']['variance_threshold']

        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        X = df[feature_cols]
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        selected = X.columns[selector.get_support()].tolist()
        self.feature_rankings['variance_threshold'] = {
            col: 1 if col in selected else 0 for col in feature_cols
        }

        logger.info(f"Variance threshold: {len(selected)}/{len(feature_cols)} features selected")
        return selected

    def mutual_information(self, df: pd.DataFrame, 
                          k: int = 100) -> List[str]:
        """Select top k features by mutual information."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        X = df[feature_cols]
        y = df[self.config['data']['target_column']]

        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)

        scores = selector.scores_
        selected = X.columns[selector.get_support()].tolist()

        self.feature_rankings['mutual_information'] = dict(zip(feature_cols, scores))

        logger.info(f"Mutual information: top {k} features selected")
        return selected

    def recursive_feature_elimination(self, df: pd.DataFrame,
                                     n_features: int = 100) -> List[str]:
        """RFE with Logistic Regression."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        X = df[feature_cols]
        y = df[self.config['data']['target_column']]

        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
        selector.fit(X, y)

        selected = X.columns[selector.support_].tolist()
        rankings = dict(zip(feature_cols, selector.ranking_))

        self.feature_rankings['rfe'] = rankings

        logger.info(f"RFE: {n_features} features selected")
        return selected

    def feature_importance(self, df: pd.DataFrame) -> List[str]:
        """Select features using Random Forest importance."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        X = df[feature_cols]
        y = df[self.config['data']['target_column']]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importances = rf.feature_importances_
        self.feature_rankings['rf_importance'] = dict(zip(feature_cols, importances))

        # Select features above mean importance
        mean_importance = np.mean(importances)
        selected = [col for col, imp in zip(feature_cols, importances) if imp > mean_importance]

        logger.info(f"RF Importance: {len(selected)} features above mean importance")
        return selected

    def shap_importance(self, df: pd.DataFrame, 
                      model: Optional[object] = None) -> List[str]:
        """Select features using SHAP values."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        X = df[feature_cols].sample(min(10000, len(df)), random_state=42)
        y = df.loc[X.index, self.config['data']['target_column']]

        if model is None:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, use class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_shap = np.abs(shap_values).mean(axis=0)
        self.feature_rankings['shap'] = dict(zip(feature_cols, mean_shap))

        # Select top 50%
        threshold = np.percentile(mean_shap, 50)
        selected = [col for col, val in zip(feature_cols, mean_shap) if val > threshold]

        logger.info(f"SHAP: {len(selected)} features selected")
        return selected

    def ensemble_selection(self, df: pd.DataFrame, 
                          methods: List[str] = None) -> List[str]:
        """
        Ensemble multiple selection methods with voting.

        Args:
            df: Input dataframe
            methods: List of methods to use
        """
        if methods is None:
            methods = ['variance_threshold', 'mutual_information', 
                      'rf_importance', 'shap']

        all_selected = {}
        for method in methods:
            if method == 'variance_threshold':
                all_selected[method] = set(self.variance_threshold(df))
            elif method == 'mutual_information':
                all_selected[method] = set(self.mutual_information(df))
            elif method == 'rfe':
                all_selected[method] = set(self.recursive_feature_elimination(df))
            elif method == 'rf_importance':
                all_selected[method] = set(self.feature_importance(df))
            elif method == 'shap':
                all_selected[method] = set(self.shap_importance(df))

        # Voting: feature must be selected by at least 2 methods
        from collections import Counter
        feature_votes = Counter()
        for selected in all_selected.values():
            for feat in selected:
                feature_votes[feat] += 1

        # Select features with >= 2 votes
        self.selected_features = [f for f, v in feature_votes.items() if v >= 2]

        # Generate ranking report
        self._generate_ranking_report(df)

        logger.info(f"Ensemble selection: {len(self.selected_features)} features selected")
        return self.selected_features

    def _generate_ranking_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature ranking report."""
        feature_cols = [c for c in df.columns if c not in 
                       [self.config['data']['id_column'], 
                        self.config['data']['target_column']]]

        report = pd.DataFrame(index=feature_cols)

        for method, scores in self.feature_rankings.items():
            report[method] = report.index.map(scores)

        # Normalize scores
        for col in report.columns:
            if report[col].max() != report[col].min():
                report[col] = (report[col] - report[col].min()) / (report[col].max() - report[col].min())

        report['ensemble_score'] = report.mean(axis=1)
        report['selected'] = report.index.isin(self.selected_features)
        report = report.sort_values('ensemble_score', ascending=False)

        self.ranking_report = report
        logger.info("Feature ranking report generated")
        return report
