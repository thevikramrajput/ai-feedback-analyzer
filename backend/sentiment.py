"""
Sentiment Analysis Module
Handles sentiment classification and analysis.
"""

import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter


def get_sentiment_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of sentiment labels.
    
    Args:
        df: DataFrame with 'sentiment' column
        
    Returns:
        Dictionary with sentiment counts
    """
    if 'sentiment' not in df.columns:
        raise ValueError("DataFrame must have 'sentiment' column")
    
    return df['sentiment'].value_counts().to_dict()


def get_sentiment_percentages(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get sentiment distribution as percentages.
    
    Args:
        df: DataFrame with 'sentiment' column
        
    Returns:
        Dictionary with sentiment percentages
    """
    distribution = get_sentiment_distribution(df)
    total = sum(distribution.values())
    
    return {k: round((v / total) * 100, 2) for k, v in distribution.items()}


def get_negative_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get all negative reviews.
    
    Args:
        df: DataFrame with reviews
        
    Returns:
        DataFrame containing only negative reviews
    """
    return df[df['sentiment'] == 'negative'].copy()


def get_sentiment_by_app(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Get sentiment distribution grouped by app.
    
    Args:
        df: DataFrame with 'sentiment' and 'app_id' columns
        
    Returns:
        Nested dictionary with sentiment counts per app
    """
    if 'app_id' not in df.columns:
        return {'all': get_sentiment_distribution(df)}
    
    result = {}
    for app_id in df['app_id'].unique():
        app_df = df[df['app_id'] == app_id]
        result[app_id] = get_sentiment_distribution(app_df)
    
    return result


def get_sentiment_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive sentiment summary.
    
    Args:
        df: DataFrame with sentiment data
        
    Returns:
        Dictionary with sentiment summary statistics
    """
    distribution = get_sentiment_distribution(df)
    percentages = get_sentiment_percentages(df)
    
    total = len(df)
    negative_count = distribution.get('negative', 0)
    
    summary = {
        'total_reviews': total,
        'distribution': distribution,
        'percentages': percentages,
        'negative_count': negative_count,
        'negative_percentage': percentages.get('negative', 0),
        'positive_count': distribution.get('positive', 0),
        'positive_percentage': percentages.get('positive', 0),
        'neutral_count': distribution.get('neutral', 0),
        'neutral_percentage': percentages.get('neutral', 0),
        'health_score': round(100 - percentages.get('negative', 0), 2)
    }
    
    return summary


def analyze_sentiment_trends(df: pd.DataFrame, date_column: str = 'at') -> Dict:
    """
    Analyze sentiment trends over time (if date column exists).
    
    Args:
        df: DataFrame with reviews
        date_column: Name of the date column
        
    Returns:
        Dictionary with trend analysis
    """
    if date_column not in df.columns:
        return {'trend': 'no_date_data'}
    
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_column])
    
    if len(df_copy) == 0:
        return {'trend': 'no_valid_dates'}
    
    # Group by date and sentiment
    df_copy['date'] = df_copy[date_column].dt.date
    daily_sentiment = df_copy.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    
    return {
        'trend': 'available',
        'daily_data': daily_sentiment.to_dict()
    }


if __name__ == "__main__":
    # Test sentiment module
    from data_loader import load_reviews, DEFAULT_DATA_PATH
    from preprocess import preprocess_reviews
    
    df = load_reviews(DEFAULT_DATA_PATH)
    processed_df = preprocess_reviews(df, filter_english=True)
    
    summary = get_sentiment_summary(processed_df)
    
    print("\n📊 Sentiment Summary:")
    print(f"  Total Reviews: {summary['total_reviews']}")
    print(f"  Positive: {summary['positive_count']} ({summary['positive_percentage']}%)")
    print(f"  Neutral: {summary['neutral_count']} ({summary['neutral_percentage']}%)")
    print(f"  Negative: {summary['negative_count']} ({summary['negative_percentage']}%)")
    print(f"  Health Score: {summary['health_score']}%")
