"""
Preprocessing Module
Handles text cleaning, filtering, and sentiment label assignment.
"""

import re
import pandas as pd
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean review text by removing noise.
    
    Args:
        text: Raw review text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def score_to_sentiment(score: int) -> str:
    """
    Convert star rating to sentiment label.
    
    Args:
        score: Star rating (1-5)
        
    Returns:
        Sentiment label: 'negative', 'neutral', or 'positive'
    """
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'


def filter_english_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to keep only English reviews.
    
    Args:
        df: DataFrame with reviews
        
    Returns:
        Filtered DataFrame with English reviews only
    """
    if 'userLang' not in df.columns:
        print("⚠ No 'userLang' column found. Keeping all reviews.")
        return df
    
    # Filter for English reviews (handling variations)
    english_mask = df['userLang'].str.strip().str.upper().isin(['EN', 'ENGLISH'])
    filtered_df = df[english_mask].copy()
    
    print(f"✓ Filtered to {len(filtered_df)} English reviews (from {len(df)} total)")
    
    return filtered_df


def preprocess_reviews(df: pd.DataFrame, filter_english: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline for review data.
    
    Args:
        df: Raw DataFrame with reviews
        filter_english: Whether to filter for English reviews only
        
    Returns:
        Preprocessed DataFrame with cleaned text and sentiment labels
    """
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Filter English reviews if requested
    if filter_english and 'userLang' in processed_df.columns:
        processed_df = filter_english_reviews(processed_df)
    
    # Remove rows with missing content
    initial_count = len(processed_df)
    processed_df = processed_df.dropna(subset=['content'])
    processed_df = processed_df[processed_df['content'].str.strip() != '']
    print(f"✓ Removed {initial_count - len(processed_df)} empty reviews")
    
    # Clean text
    processed_df['cleaned_content'] = processed_df['content'].apply(clean_text)
    
    # Remove rows where cleaned content is empty
    processed_df = processed_df[processed_df['cleaned_content'].str.strip() != '']
    
    # Add sentiment labels based on score
    if 'score' in processed_df.columns:
        processed_df['sentiment'] = processed_df['score'].apply(score_to_sentiment)
        
        # Print sentiment distribution
        sentiment_counts = processed_df['sentiment'].value_counts()
        print(f"✓ Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(processed_df)) * 100
            print(f"    {sentiment}: {count} ({pct:.1f}%)")
    
    # Reset index
    processed_df = processed_df.reset_index(drop=True)
    
    print(f"✓ Preprocessing complete. {len(processed_df)} reviews ready for analysis.")
    
    return processed_df


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_reviews, DEFAULT_DATA_PATH
    
    df = load_reviews(DEFAULT_DATA_PATH)
    processed_df = preprocess_reviews(df, filter_english=True)
    
    print("\n📝 Sample preprocessed reviews:")
    for i, row in processed_df.head(3).iterrows():
        print(f"\n  Original: {row['content'][:80]}...")
        print(f"  Cleaned:  {row['cleaned_content'][:80]}...")
        print(f"  Score: {row['score']} → Sentiment: {row['sentiment']}")
