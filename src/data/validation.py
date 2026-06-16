"""
Data Validation Module
Schema validation, business rules, and data contracts
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml

logger = logging.getLogger("banking_ml.data.validation")


@dataclass
class ValidationReport:
    """Data validation report structure."""
    passed: bool
    checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]


class DataValidator:
    """
    Production-grade data validator.

    Implements:
    - Schema validation
    - Business rule checks
    - Data quality metrics
    - Great Expectations-style validation
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("DataValidator initialized")

    def validate_schema(self, df: pd.DataFrame) -> ValidationReport:
        """Validate dataframe schema against expected structure."""
        checks = {}
        errors = []
        warnings = []

        # Check required columns
        required_cols = [self.config['data']['target_column'], 
                        self.config['data']['id_column']]
        for col in required_cols:
            checks[f'has_{col}'] = col in df.columns
            if not checks[f'has_{col}']:
                errors.append(f"Missing required column: {col}")

        # Check feature count
        feature_cols = [c for c in df.columns if c not in required_cols]
        checks['feature_count'] = len(feature_cols) == self.config['data']['n_features']
        if not checks['feature_count']:
            warnings.append(f"Expected {self.config['data']['n_features']} features, got {len(feature_cols)}")

        # Check target values
        if self.config['data']['target_column'] in df.columns:
            target_vals = set(df[self.config['data']['target_column']].unique())
            checks['target_binary'] = target_vals.issubset({0, 1})
            if not checks['target_binary']:
                errors.append(f"Target column contains non-binary values: {target_vals}")

        # Check for empty dataframe
        checks['not_empty'] = len(df) > 0
        if not checks['not_empty']:
            errors.append("Dataframe is empty")

        # Check ID uniqueness
        if self.config['data']['id_column'] in df.columns:
            checks['id_unique'] = df[self.config['data']['id_column']].is_unique
            if not checks['id_unique']:
                errors.append("ID column contains duplicates")

        passed = all(checks.values()) and len(errors) == 0

        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'features': len(feature_cols),
            'target_distribution': df[self.config['data']['target_column']].value_counts().to_dict() 
                                if self.config['data']['target_column'] in df.columns else None
        }

        report = ValidationReport(passed, checks, errors, warnings, stats)

        if passed:
            logger.info("Schema validation passed")
        else:
            logger.error(f"Schema validation failed: {errors}")

        return report

    def validate_data_quality(self, df: pd.DataFrame) -> ValidationReport:
        """Validate data quality metrics."""
        checks = {}
        errors = []
        warnings = []

        # Missing value check
        missing_pct = df.isnull().mean().max()
        checks['missing_below_threshold'] = missing_pct < self.config['data']['missing_threshold']
        if not checks['missing_below_threshold']:
            warnings.append(f"Missing values exceed threshold: {missing_pct:.2%}")

        # Duplicate check
        dup_count = df.duplicated().sum()
        checks['no_duplicates'] = dup_count == 0
        if not checks['no_duplicates']:
            warnings.append(f"Found {dup_count} duplicate rows")

        # Numeric feature check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        checks['all_numeric_features'] = len(numeric_cols) >= self.config['data']['n_features']

        # Infinity check
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        checks['no_infinity'] = inf_count == 0
        if not checks['no_infinity']:
            errors.append(f"Found {inf_count} infinite values")

        passed = all(checks.values()) and len(errors) == 0

        stats = {
            'missing_pct': missing_pct,
            'duplicate_count': dup_count,
            'infinite_count': inf_count,
            'numeric_features': len(numeric_cols)
        }

        return ValidationReport(passed, checks, errors, warnings, stats)
