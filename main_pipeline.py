#!/usr/bin/env python3
"""
Customer Transaction Prediction - Main Pipeline
Orchestrates all phases from data ingestion to model deployment

Usage:
    python main_pipeline.py --phase all --data data/raw/train.csv
"""

import argparse
import sys
import os
import logging
import json
from datetime import datetime

sys.path.insert(0, 'src')

from data.extract import DataExtractor
from data.transform import DataTransformer
from data.validation import DataValidator
from data.quality_checks import QualityChecker
from features.builders import FeatureBuilder
from features.selectors import FeatureSelector
from models.train import ModelTrainer
from models.evaluate import ModelEvaluator
from models.interpret import ModelInterpreter
from pipeline.versioning import ModelVersioning

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_all_phases(data_path):
    """Run complete end-to-end pipeline"""
    logger.info("="*80)
    logger.info("STARTING COMPLETE PIPELINE")
    logger.info("="*80)

    start_time = datetime.now()

    # Phase 2: EDA
    extractor = DataExtractor()
    df = extractor.extract_csv(data_path)
    logger.info(f"Data loaded: {df.shape}")

    # Phase 3: Feature Engineering
    builder = FeatureBuilder()
    df = builder.build_all_features(df)

    # Phase 4: Feature Selection
    selector = FeatureSelector()
    selected = selector.ensemble_selection(df)

    # Prepare data
    feature_cols = [c for c in df.columns if c not in ['ID_code', 'target']]
    X = df[feature_cols]
    y = df['target']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Phase 5: Model Training
    trainer = ModelTrainer()
    cv_results = trainer.train_with_cv(X_train, y_train)

    # Phase 6: Evaluation
    evaluator = ModelEvaluator()
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['cv_mean_auc'] if cv_results[k]['cv_mean_auc'] else 0)
    best_model = cv_results[best_model_name]['model']

    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    report = evaluator.generate_evaluation_report(y_test.values, y_prob, y_pred, best_model_name)

    # Phase 7: Interpretability
    interpreter = ModelInterpreter()
    interpreter.generate_interpretability_report(best_model, X_test, y_test)

    # Phase 8: MLOps
    versioning = ModelVersioning()
    metrics = report['metrics']
    model_path = trainer.save_model(best_model, best_model_name)
    versioning.register_model(best_model_name, "1.0.0", metrics, 
                              best_model.get_params(), model_path)
    versioning.promote_to_production(best_model_name, "1.0.0")

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"PIPELINE COMPLETE - Duration: {duration:.2f}s")

    return {'status': 'success', 'duration': duration, 'best_model': best_model_name}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all')
    parser.add_argument('--data', default='data/raw/train.csv')
    args = parser.parse_args()

    result = run_all_phases(args.data)
    print(json.dumps(result, indent=2, default=str))
