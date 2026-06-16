"""
End-to-End ML Pipeline
Reproducible workflow with logging and artifact tracking
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
import yaml
import pickle
import json
from datetime import datetime
import os

# Import project modules
from ..data.extract import DataExtractor
from ..data.transform import DataTransformer
from ..data.validation import DataValidator
from ..data.quality_checks import QualityChecker
from ..data.load import DataLoader
from ..features.builders import FeatureBuilder
from ..features.transformers import FeatureTransformer
from ..features.selectors import FeatureSelector
from ..models.train import ModelTrainer
from ..models.evaluate import ModelEvaluator
from ..models.tune import HyperparameterTuner
from ..models.interpret import ModelInterpreter

logger = logging.getLogger("banking_ml.pipeline")


class MLPipeline:
    """
    Production end-to-end ML pipeline.

    Pipeline stages:
    1. Data Extraction & Validation
    2. Data Quality Checks
    3. Data Transformation
    4. Feature Engineering
    5. Feature Selection
    6. Model Training & CV
    7. Hyperparameter Tuning
    8. Model Evaluation
    9. Model Interpretation
    10. Model Saving & Versioning
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.artifacts = {}
        self.metrics = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize components
        self.extractor = DataExtractor(config_path)
        self.validator = DataValidator(config_path)
        self.quality_checker = QualityChecker(config_path)
        self.transformer = DataTransformer(config_path)
        self.loader = DataLoader(config_path)
        self.feature_builder = FeatureBuilder(config_path)
        self.feature_transformer = FeatureTransformer(config_path)
        self.feature_selector = FeatureSelector(config_path)
        self.model_trainer = ModelTrainer(config_path)
        self.model_evaluator = ModelEvaluator(config_path)
        self.hyper_tuner = HyperparameterTuner(config_path)
        self.model_interpreter = ModelInterpreter(config_path)

        logger.info(f"MLPipeline initialized - Run ID: {self.run_id}")

    def run(self, data_path: str, target_model: str = 'LightGBM',
           tune_model: bool = True) -> Dict:
        """
        Execute complete pipeline.

        Args:
            data_path: Path to raw data
            target_model: Primary model to train and tune
            tune_model: Whether to perform hyperparameter tuning
        """
        logger.info("=" * 60)
        logger.info("STARTING ML PIPELINE")
        logger.info("=" * 60)

        # Stage 1: Data Extraction
        logger.info("STAGE 1: Data Extraction")
        df = self.extractor.extract_csv(data_path)
        self.artifacts['raw_data_shape'] = df.shape

        # Stage 2: Data Validation
        logger.info("STAGE 2: Data Validation")
        validation_report = self.validator.validate_schema(df)
        if not validation_report.passed:
            logger.error("Data validation failed! Aborting pipeline.")
            return {'status': 'failed', 'errors': validation_report.errors}

        quality_report = self.validator.validate_data_quality(df)
        self.artifacts['validation'] = {
            'schema': validation_report.__dict__,
            'quality': quality_report.__dict__
        }

        # Stage 3: Quality Analysis
        logger.info("STAGE 3: Quality Analysis")
        quality_metrics = self.quality_checker.generate_quality_report(df)
        distribution_analysis = self.quality_checker.check_distribution(df)
        self.artifacts['quality_metrics'] = quality_metrics
        self.artifacts['distribution_analysis'] = distribution_analysis

        # Stage 4: Data Transformation
        logger.info("STAGE 4: Data Transformation")
        df = self.transformer.handle_missing(df, strategy='median')
        df = self.transformer.treat_outliers(df, method='iqr')
        df = self.transformer.optimize_dtypes(df)

        # Stage 5: Feature Engineering
        logger.info("STAGE 5: Feature Engineering")
        df = self.feature_builder.build_all_features(df)
        self.artifacts['engineered_features'] = len(df.columns)

        # Stage 6: Feature Selection
        logger.info("STAGE 6: Feature Selection")
        selected_features = self.feature_selector.ensemble_selection(df)

        # Prepare final dataset
        id_col = self.config['data']['id_column']
        target_col = self.config['data']['target_column']
        final_cols = [id_col, target_col] + selected_features
        df_final = df[final_cols]

        # Split data
        X = df_final.drop(columns=[id_col, target_col])
        y = df_final[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )

        # Stage 7: Feature Scaling
        logger.info("STAGE 7: Feature Scaling")
        X_train_scaled = self.feature_transformer.fit_transform(X_train, 'robust')
        X_test_scaled = self.feature_transformer.transform(X_test, 'robust')

        # Stage 8: Model Training
        logger.info("STAGE 8: Model Training")
        cv_results = self.model_trainer.train_with_cv(X_train_scaled, y_train)
        self.metrics['cv_results'] = cv_results

        # Stage 9: Hyperparameter Tuning
        if tune_model:
            logger.info("STAGE 9: Hyperparameter Tuning")
            tune_results = self.hyper_tuner.optuna_optimize(
                X_train_scaled, y_train, target_model,
                n_trials=self.config['modeling']['n_trials_optuna']
            )
            self.metrics['tuning'] = tune_results

            # Retrain with best params
            best_model = self.model_trainer.train_single_model(
                X_train_scaled, y_train, target_model, tune_results['best_params']
            )
        else:
            best_model = self.model_trainer.trained_models.get(target_model)

        # Stage 10: Model Evaluation
        logger.info("STAGE 10: Model Evaluation")
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        eval_report = self.model_evaluator.generate_evaluation_report(
            y_test.values, y_prob, y_pred, target_model
        )
        self.metrics['evaluation'] = eval_report

        # Stage 11: Model Interpretation
        logger.info("STAGE 11: Model Interpretation")
        interpret_report = self.model_interpreter.generate_interpretability_report(
            best_model, X_test_scaled, y_test
        )
        self.metrics['interpretability'] = interpret_report

        # Stage 12: Save Artifacts
        logger.info("STAGE 12: Saving Artifacts")
        self._save_artifacts(best_model, X_train_scaled, y_train)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return {
            'status': 'success',
            'run_id': self.run_id,
            'artifacts': self.artifacts,
            'metrics': self.metrics,
            'best_model': best_model
        }

    def _save_artifacts(self, model, X_train, y_train):
        """Save all pipeline artifacts."""
        artifact_dir = f"models/artifacts/{self.run_id}"
        os.makedirs(artifact_dir, exist_ok=True)

        # Save model
        model_path = self.model_trainer.save_model(model, 'best_model', 
                                                   f"{artifact_dir}/model.pkl")

        # Save feature selector
        with open(f"{artifact_dir}/feature_selector.pkl", 'wb') as f:
            pickle.dump(self.feature_selector, f)

        # Save scaler
        with open(f"{artifact_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_transformer.scalers, f)

        # Save metrics
        with open(f"{artifact_dir}/metrics.json", 'w') as f:
            json.dump({
                'run_id': self.run_id,
                'artifacts': {k: str(v) for k, v in self.artifacts.items()},
                'metrics_summary': {
                    'cv_auc': self.metrics.get('cv_results', {}).get('LightGBM', {}).get('cv_mean_auc'),
                    'test_auc': self.metrics.get('evaluation', {}).get('metrics', {}).get('roc_auc')
                }
            }, f, indent=2, default=str)

        logger.info(f"Artifacts saved to {artifact_dir}")
