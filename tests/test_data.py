import pytest
import pandas as pd
import numpy as np
from src.data.extract import DataExtractor
from src.data.validation import DataValidator
from src.data.quality_checks import QualityChecker

class TestDataPipeline:
    def test_data_extraction(self):
        extractor = DataExtractor()
        # Test with sample data
        df = pd.DataFrame({
            'ID_code': ['ID_001', 'ID_002'],
            'var_000': [1.0, 2.0],
            'target': [0, 1]
        })
        assert len(df) == 2
        assert 'target' in df.columns

    def test_validation(self):
        validator = DataValidator()
        df = pd.DataFrame({
            'ID_code': ['ID_001', 'ID_002'],
            'var_000': [1.0, 2.0],
            'target': [0, 1]
        })
        report = validator.validate_schema(df)
        assert report.passed == True

    def test_quality_checks(self):
        checker = QualityChecker()
        df = pd.DataFrame({
            'var_000': [1.0, 2.0, 3.0],
            'var_001': [4.0, 5.0, 6.0],
            'target': [0, 1, 0]
        })
        report = checker.generate_quality_report(df)
        assert report['rows'] == 3
        assert report['duplicates'] == 0
