"""
Clustering Module
Handles topic discovery using embeddings and KMeans clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re


class TopicClusterer:
    """
    Topic clustering using TF-IDF vectorization and KMeans.
    Falls back to TF-IDF if sentence-transformers is not available.
    """
    
    def __init__(self, n_clusters: int = 8, use_embeddings: bool = True):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of topic clusters to create
            use_embeddings: Whether to try using sentence-transformers
        """
        self.n_clusters = n_clusters
        self.use_embeddings = use_embeddings
        self.embedder = None
        self.kmeans = None
        self.vectorizer = None
        self.cluster_labels = {}
        
        # Try to load sentence-transformers
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Using SentenceTransformer for embeddings")
            except ImportError:
                print("⚠ sentence-transformers not available, falling back to TF-IDF")
                self.use_embeddings = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings or TF-IDF vectors for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of vectors
        """
        if self.use_embeddings and self.embedder:
            print(f"  Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            return embeddings
        else:
            # Fall back to TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            vectors = self.vectorizer.fit_transform(texts)
            return vectors.toarray()
    
    def cluster(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Perform clustering on texts.
        
        Args:
            texts: List of text strings to cluster
            
        Returns:
            Tuple of (cluster_labels, cluster_info)
        """
        if len(texts) < self.n_clusters:
            self.n_clusters = max(2, len(texts) // 2)
            print(f"⚠ Reduced clusters to {self.n_clusters} due to small dataset")
        
        # Get vectors
        vectors = self.fit_transform(texts)
        
        # Perform KMeans clustering
        print(f"  Running KMeans with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(vectors)
        
        # Generate cluster info
        cluster_info = self._generate_cluster_info(texts, labels)
        
        return labels, cluster_info
    
    def _generate_cluster_info(self, texts: List[str], labels: np.ndarray) -> Dict:
        """
        Generate descriptive information about each cluster.
        
        Args:
            texts: Original texts
            labels: Cluster labels
            
        Returns:
            Dictionary with cluster information
        """
        cluster_info = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
            
            if not cluster_texts:
                continue
            
            # Extract keywords using simple frequency analysis
            keywords = self._extract_keywords(cluster_texts)
            
            # Generate cluster label from top keywords
            label = " / ".join(keywords[:3]) if keywords else f"Topic {cluster_id}"
            
            cluster_info[cluster_id] = {
                'label': label,
                'keywords': keywords,
                'count': len(cluster_texts),
                'sample_texts': cluster_texts[:3]
            }
            
            self.cluster_labels[cluster_id] = label
        
        return cluster_info
    
    def _extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        """
        Extract top keywords from a list of texts.
        
        Args:
            texts: List of texts
            top_n: Number of keywords to extract
            
        Returns:
            List of top keywords
        """
        # Common stopwords to filter out
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'it', 'its',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'you', 'your', 'we',
            'our', 'they', 'their', 'he', 'she', 'him', 'her', 'very', 'just',
            'app', 'application', 'use', 'using', 'good', 'bad', 'like', 'dont',
            'not', 'no', 'yes', 'get', 'got', 'one', 'two', 'new', 'more', 'also'
        }
        
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            words = [w for w in words if w not in stopwords]
            word_counts.update(words)
        
        # Get top keywords
        top_keywords = [word for word, _ in word_counts.most_common(top_n)]
        
        return top_keywords


def cluster_reviews(df: pd.DataFrame, n_clusters: int = 8, 
                    text_column: str = 'cleaned_content',
                    focus_negative: bool = True) -> pd.DataFrame:
    """
    Add topic clusters to review DataFrame.
    
    Args:
        df: DataFrame with reviews
        n_clusters: Number of clusters
        text_column: Column containing text to cluster
        focus_negative: Whether to only cluster negative reviews
        
    Returns:
        DataFrame with added cluster columns
    """
    df_result = df.copy()
    
    # Optionally focus on negative reviews for complaint discovery
    if focus_negative and 'sentiment' in df.columns:
        mask = df['sentiment'] == 'negative'
        texts_to_cluster = df.loc[mask, text_column].tolist()
        indices = df.loc[mask].index.tolist()
        print(f"✓ Clustering {len(texts_to_cluster)} negative reviews")
    else:
        texts_to_cluster = df[text_column].tolist()
        indices = df.index.tolist()
        print(f"✓ Clustering all {len(texts_to_cluster)} reviews")
    
    if len(texts_to_cluster) < 2:
        print("⚠ Not enough reviews to cluster")
        df_result['cluster_id'] = -1
        df_result['cluster_label'] = 'Unclustered'
        return df_result
    
    # Initialize clusterer
    clusterer = TopicClusterer(n_clusters=n_clusters, use_embeddings=True)
    
    # Perform clustering
    labels, cluster_info = clusterer.cluster(texts_to_cluster)
    
    # Add cluster information to DataFrame
    df_result['cluster_id'] = -1
    df_result['cluster_label'] = 'Other'
    
    for i, idx in enumerate(indices):
        cluster_id = labels[i]
        df_result.loc[idx, 'cluster_id'] = cluster_id
        df_result.loc[idx, 'cluster_label'] = cluster_info.get(cluster_id, {}).get('label', f'Topic {cluster_id}')
    
    print(f"✓ Clustering complete. {len(cluster_info)} topics discovered.")
    
    return df_result, cluster_info


def get_top_issues(cluster_info: Dict, top_n: int = 5) -> List[Dict]:
    """
    Get the top issues (largest clusters) sorted by count.
    
    Args:
        cluster_info: Dictionary with cluster information
        top_n: Number of top issues to return
        
    Returns:
        List of top issues with their details
    """
    issues = []
    for cluster_id, info in cluster_info.items():
        issues.append({
            'cluster_id': cluster_id,
            'label': info['label'],
            'count': info['count'],
            'keywords': info['keywords'],
            'samples': info['sample_texts'][:2]
        })
    
    # Sort by count descending
    issues.sort(key=lambda x: x['count'], reverse=True)
    
    return issues[:top_n]


if __name__ == "__main__":
    # Test clustering module
    from data_loader import load_reviews, DEFAULT_DATA_PATH
    from preprocess import preprocess_reviews
    
    df = load_reviews(DEFAULT_DATA_PATH)
    processed_df = preprocess_reviews(df, filter_english=True)
    
    # Cluster the reviews
    clustered_df, cluster_info = cluster_reviews(processed_df, n_clusters=6)
    
    print("\n🎯 Top Issues Discovered:")
    top_issues = get_top_issues(cluster_info)
    for i, issue in enumerate(top_issues, 1):
        print(f"\n  {i}. {issue['label']}")
        print(f"     Count: {issue['count']} complaints")
        print(f"     Keywords: {', '.join(issue['keywords'])}")
        if issue['samples']:
            print(f"     Sample: \"{issue['samples'][0][:80]}...\"")
