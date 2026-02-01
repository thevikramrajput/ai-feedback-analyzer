"""
Data Loader Module
Handles loading and initial validation of CSV review datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_reviews(file_path: str) -> pd.DataFrame:
    """
    Load reviews from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing reviews
        
    Returns:
        DataFrame with review data
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    # Try different encodings to handle various file formats
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✓ Successfully loaded {len(df)} reviews using {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Unable to decode file with any supported encoding: {encodings}")


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset contains required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = ['content', 'score']
    optional_columns = ['userLang', 'app_id', 'userName', 'at']
    
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    present_optional = [col for col in optional_columns if col in df.columns]
    print(f"✓ Dataset validated. Optional columns present: {present_optional}")
    
    return True


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame with review data
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'total_reviews': len(df),
        'columns': list(df.columns),
        'score_distribution': df['score'].value_counts().to_dict() if 'score' in df.columns else {},
        'languages': df['userLang'].value_counts().to_dict() if 'userLang' in df.columns else {},
        'apps': df['app_id'].unique().tolist() if 'app_id' in df.columns else [],
        'missing_content': df['content'].isna().sum() if 'content' in df.columns else 0
    }
    
    return info


# Default dataset path
DEFAULT_DATA_PATH = r"c:\Users\vikra\OneDrive - BENNETT UNIVERSITY\Sem 6\Machine Learning\Product Review analyzer\data\Training_Data.csv"


if __name__ == "__main__":
    # Test the data loader
    df = load_reviews(DEFAULT_DATA_PATH)
    validate_dataset(df)
    info = get_dataset_info(df)
    
    print("\n📊 Dataset Info:")
    print(f"  Total Reviews: {info['total_reviews']}")
    print(f"  Score Distribution: {info['score_distribution']}")
    print(f"  Languages: {len(info['languages'])} unique")
    print(f"  Apps: {info['apps']}")
