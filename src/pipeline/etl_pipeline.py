import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BankingETLPipeline:
    """Production ETL pipeline with validation and quality checks."""

    def __init__(self, config=None):
        self.config = config or {
            'expected_columns': 200,
            'target_column': 'target',
            'id_column': 'ID_code',
            'max_missing_pct': 10,
            'min_rows': 1000,
            'outlier_threshold': 4.0
        }
        self.validation_results = []
        self.processing_stats = {}

    def extract(self, source_path):
        """Extract data from source."""
        logger.info(f"Extracting from {source_path}")
        start = datetime.now()

        if source_path.endswith('.csv'):
            data = pd.read_csv(source_path)
        elif source_path.endswith('.parquet'):
            data = pd.read_parquet(source_path)
        else:
            raise ValueError(f"Unsupported format: {source_path}")

        self.processing_stats['extraction'] = {
            'rows': len(data), 'columns': len(data.columns),
            'time': (datetime.now() - start).total_seconds()
        }
        return data

    def validate(self, data):
        """Validate data quality."""
        logger.info("Validating data quality...")
        checks = []

        feature_cols = [c for c in data.columns if c not in [self.config['id_column'], self.config['target_column']]]

        checks.append({
            'check': 'Row Count', 'expected': f">= {self.config['min_rows']}",
            'actual': len(data), 'status': 'PASS' if len(data) >= self.config['min_rows'] else 'FAIL'
        })

        checks.append({
            'check': 'Feature Count', 'expected': self.config['expected_columns'],
            'actual': len(feature_cols), 'status': 'PASS' if len(feature_cols) == self.config['expected_columns'] else 'FAIL'
        })

        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        checks.append({
            'check': 'Missing Values', 'expected': f"<= {self.config['max_missing_pct']}%",
            'actual': f"{missing_pct:.2f}%", 'status': 'PASS' if missing_pct <= self.config['max_missing_pct'] else 'WARN'
        })

        if self.config['target_column'] in data.columns:
            target_balance = data[self.config['target_column']].mean()
            checks.append({
                'check': 'Target Balance', 'expected': '0.01 - 0.99',
                'actual': f"{target_balance:.4f}", 'status': 'PASS' if 0.01 <= target_balance <= 0.99 else 'WARN'
            })

        dup_count = data.duplicated().sum()
        checks.append({
            'check': 'Duplicates', 'expected': '0', 'actual': dup_count,
            'status': 'PASS' if dup_count == 0 else 'WARN'
        })

        self.validation_results = checks
        passed = sum(1 for c in checks if c['status'] == 'PASS')
        failed = sum(1 for c in checks if c['status'] == 'FAIL')
        logger.info(f"Validation: {passed} passed, {failed} failed")

        if failed > 0:
            raise ValueError(f"Validation failed: {[c for c in checks if c['status'] == 'FAIL']}")

        return checks

    def transform(self, data):
        """Transform data for modeling."""
        logger.info("Transforming data...")
        start = datetime.now()

        feature_cols = [c for c in data.columns if c not in [self.config['id_column'], self.config['target_column']]]
        features = data[feature_cols].copy()

        # Impute missing values
        missing_before = features.isnull().sum().sum()
        features = features.fillna(features.median())

        # Handle infinities
        features = features.replace([np.inf, -np.inf], np.nan).fillna(features.median())

        # Clip outliers
        for col in features.columns:
            q99 = features[col].quantile(0.99)
            q01 = features[col].quantile(0.01)
            features[col] = features[col].clip(lower=q01, upper=q99)

        self.processing_stats['transformation'] = {
            'features': len(features.columns),
            'missing_imputed': int(missing_before),
            'time': (datetime.now() - start).total_seconds()
        }

        result = features
        if self.config['id_column'] in data.columns:
            result[self.config['id_column']] = data[self.config['id_column']]
        if self.config['target_column'] in data.columns:
            result[self.config['target_column']] = data[self.config['target_column']]

        return result

    def load(self, data, output_path, format='parquet'):
        """Load to destination."""
        logger.info(f"Loading to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == 'parquet':
            data.to_parquet(output_path, index=False)
        elif format == 'csv':
            data.to_csv(output_path, index=False)

        return output_path

    def run_pipeline(self, source_path, output_path):
        """Execute full ETL."""
        logger.info("="*60)
        logger.info("STARTING ETL PIPELINE")
        logger.info("="*60)

        overall_start = datetime.now()

        data = self.extract(source_path)
        self.validate(data)
        transformed = self.transform(data)
        self.load(transformed, output_path)

        overall_time = (datetime.now() - overall_start).total_seconds()

        summary = {
            'status': 'SUCCESS',
            'total_time': overall_time,
            'extraction': self.processing_stats['extraction'],
            'validation': self.validation_results,
            'transformation': self.processing_stats['transformation']
        }

        logger.info(f"ETL Complete: {overall_time:.2f}s")
        return summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    pipeline = BankingETLPipeline()
    result = pipeline.run_pipeline(args.input, args.output)
    print(json.dumps(result, indent=2, default=str))
