import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

class TestModelPipeline:
    def test_model_training(self):
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        trainer = ModelTrainer()

        # Test with subset of models for speed
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }

        results = trainer.train_with_cv(X, y, models=models, cv_folds=3)
        assert len(results) == 2
        assert all(r['cv_mean_auc'] is not None for r in results.values())

    def test_evaluation(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
        y_pred = (y_prob > 0.5).astype(int)

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        assert 'roc_auc' in metrics
        assert 'business' in metrics
        assert metrics['roc_auc'] > 0
