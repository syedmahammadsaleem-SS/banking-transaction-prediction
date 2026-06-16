"""
Model Versioning Module
MLflow-style model tracking and registry
"""

import logging
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("banking_ml.pipeline.versioning")


class ModelVersioning:
    """
    Simple model versioning system (MLflow alternative for production).

    Tracks:
    - Model versions
    - Performance metrics
    - Training parameters
    - Data lineage
    """

    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.registry_file = f"{registry_path}/registry.json"
        self.registry = self._load_registry()
        logger.info("ModelVersioning initialized")

    def _load_registry(self) -> Dict:
        """Load existing registry."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': [], 'current_production': None}

    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(self, model_name: str, version: str,
                      metrics: Dict, params: Dict,
                      artifact_path: str,
                      tags: Optional[Dict] = None) -> Dict:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            version: Version string (e.g., '1.0.0')
            metrics: Performance metrics
            params: Training parameters
            artifact_path: Path to saved model
            tags: Optional tags
        """
        model_entry = {
            'model_name': model_name,
            'version': version,
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'params': params,
            'artifact_path': artifact_path,
            'tags': tags or {},
            'stage': 'staging'  # staging, production, archived
        }

        self.registry['models'].append(model_entry)
        self._save_registry()

        logger.info(f"Registered model {model_name} v{version}")
        return model_entry

    def promote_to_production(self, model_name: str, version: str) -> Dict:
        """Promote a model version to production."""
        for model in self.registry['models']:
            if model['model_name'] == model_name and model['version'] == version:
                # Demote current production
                if self.registry['current_production']:
                    for m in self.registry['models']:
                        if m['model_name'] == self.registry['current_production']:
                            m['stage'] = 'archived'

                model['stage'] = 'production'
                self.registry['current_production'] = f"{model_name}:{version}"
                self._save_registry()

                logger.info(f"Promoted {model_name} v{version} to production")
                return model

        raise ValueError(f"Model {model_name} v{version} not found")

    def get_production_model(self) -> Optional[Dict]:
        """Get current production model info."""
        if not self.registry['current_production']:
            return None

        name, version = self.registry['current_production'].split(':')
        for model in self.registry['models']:
            if model['model_name'] == name and model['version'] == version:
                return model
        return None

    def list_models(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """List all registered models."""
        models = self.registry['models']
        if model_name:
            models = [m for m in models if m['model_name'] == model_name]

        return pd.DataFrame(models)

    def compare_versions(self, model_name: str) -> pd.DataFrame:
        """Compare all versions of a model."""
        models = [m for m in self.registry['models'] if m['model_name'] == model_name]

        comparison = []
        for m in models:
            comparison.append({
                'version': m['version'],
                'stage': m['stage'],
                'registered_at': m['registered_at'],
                'auc': m['metrics'].get('roc_auc', 'N/A'),
                'f1': m['metrics'].get('f1', 'N/A'),
                'precision': m['metrics'].get('precision', 'N/A'),
                'recall': m['metrics'].get('recall', 'N/A')
            })

        return pd.DataFrame(comparison)
