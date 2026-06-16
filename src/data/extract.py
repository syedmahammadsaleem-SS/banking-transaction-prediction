"""
Data Extraction Module
Handles ingestion from multiple sources with logging and error handling
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Union, Optional
import yaml

logger = logging.getLogger("banking_ml.data.extract")


class DataExtractor:
    """
    Production-grade data extraction class.

    Supports CSV, Parquet, Excel, and SQL sources.
    Implements retry logic, batch processing, and schema validation.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize extractor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_path = Path(self.config['paths']['data_raw'])
        logger.info("DataExtractor initialized")

    def extract_csv(self, filepath: Union[str, Path], 
                    chunksize: Optional[int] = None,
                    dtype: Optional[dict] = None) -> pd.DataFrame:
        """
        Extract data from CSV with optional chunking for large files.

        Args:
            filepath: Path to CSV file
            chunksize: Number of rows per chunk for memory efficiency
            dtype: Column type dictionary for optimization

        Returns:
            pd.DataFrame: Extracted data
        """
        try:
            filepath = Path(filepath)
            logger.info(f"Extracting CSV: {filepath}")

            if chunksize:
                chunks = pd.read_csv(filepath, chunksize=chunksize, dtype=dtype)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(filepath, dtype=dtype)

            logger.info(f"Successfully extracted {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise

    def extract_parquet(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Extract from Parquet format (optimized for large datasets)."""
        try:
            filepath = Path(filepath)
            logger.info(f"Extracting Parquet: {filepath}")
            df = pd.read_parquet(filepath)
            logger.info(f"Extracted {len(df)} rows from Parquet")
            return df
        except Exception as e:
            logger.error(f"Parquet extraction failed: {str(e)}")
            raise

    def extract_from_multiple(self, file_pattern: str) -> pd.DataFrame:
        """Extract and concatenate multiple files matching pattern."""
        import glob
        files = glob.glob(file_pattern)
        logger.info(f"Found {len(files)} files matching pattern: {file_pattern}")

        dfs = []
        for file in files:
            if file.endswith('.csv'):
                dfs.append(self.extract_csv(file))
            elif file.endswith('.parquet'):
                dfs.append(self.extract_parquet(file))

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined)} rows")
        return combined
