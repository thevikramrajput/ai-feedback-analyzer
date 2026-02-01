"""
Backend Package Initialization
AI Product Feedback Analyzer
"""

from .data_loader import load_reviews, validate_dataset, get_dataset_info, DEFAULT_DATA_PATH
from .preprocess import preprocess_reviews, clean_text, score_to_sentiment
from .sentiment import get_sentiment_summary, get_sentiment_distribution
from .clustering import cluster_reviews, get_top_issues, TopicClusterer
from .insights import generate_insights, format_insights_report

__all__ = [
    'load_reviews',
    'validate_dataset', 
    'get_dataset_info',
    'DEFAULT_DATA_PATH',
    'preprocess_reviews',
    'clean_text',
    'score_to_sentiment',
    'get_sentiment_summary',
    'get_sentiment_distribution',
    'cluster_reviews',
    'get_top_issues',
    'TopicClusterer',
    'generate_insights',
    'format_insights_report'
]
