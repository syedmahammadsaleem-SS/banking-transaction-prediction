"""
Data Loading Module
Handles data persistence and versioning
"""

import logging
import pandas as pd
from pathlib import Path
import pickle
import yaml

logger = logging.getLogger("banking_ml.data.load")


class DataLoader:
    """Production data loader with versioning support."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.processed_path = Path(self.config['paths']['data_processed'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        logger.info("DataLoader initialized")

    def save_processed(self, df: pd.DataFrame, filename: str, 
                       format: str = "parquet") -> str:
        """Save processed data with metadata."""
        filepath = self.processed_path / f"{filename}.{format}"

        if format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "pickle":
            df.to_pickle(filepath)

        # Save metadata
        metadata = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'rows': len(df)
        }

        meta_path = self.processed_path / f"{filename}_metadata.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved processed data: {filepath} ({df.shape})")
        return str(filepath)

    def load_processed(self, filename: str, format: str = "parquet") -> pd.DataFrame:
        """Load processed data."""
        filepath = self.processed_path / f"{filename}.{format}"

        if format == "parquet":
            return pd.read_parquet(filepath)
        elif format == "csv":
            return pd.read_csv(filepath)
        elif format == "pickle":
            return pd.read_pickle(filepath)
