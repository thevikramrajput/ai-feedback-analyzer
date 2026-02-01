"""
Insights Module
Generates actionable product insights from analyzed review data.
"""

import pandas as pd
from typing import Dict, List, Optional
from collections import Counter


def generate_insights(df: pd.DataFrame, cluster_info: Dict) -> Dict:
    """
    Generate comprehensive product insights from review data.
    
    Args:
        df: DataFrame with processed and clustered reviews
        cluster_info: Dictionary with cluster information
        
    Returns:
        Dictionary with all insights
    """
    insights = {
        'summary': generate_summary(df),
        'sentiment': get_sentiment_insights(df),
        'top_issues': get_top_issues_insights(cluster_info),
        'sample_complaints': get_sample_complaints(df),
        'recommendations': generate_recommendations(df, cluster_info),
        'app_breakdown': get_app_breakdown(df) if 'app_id' in df.columns else None
    }
    
    return insights


def generate_summary(df: pd.DataFrame) -> Dict:
    """
    Generate high-level summary statistics.
    
    Args:
        df: DataFrame with reviews
        
    Returns:
        Dictionary with summary stats
    """
    total = len(df)
    
    if 'sentiment' in df.columns:
        negative = len(df[df['sentiment'] == 'negative'])
        positive = len(df[df['sentiment'] == 'positive'])
        neutral = len(df[df['sentiment'] == 'neutral'])
    else:
        negative = positive = neutral = 0
    
    return {
        'total_reviews': total,
        'negative_count': negative,
        'positive_count': positive,
        'neutral_count': neutral,
        'negative_percentage': round((negative / total) * 100, 1) if total > 0 else 0,
        'positive_percentage': round((positive / total) * 100, 1) if total > 0 else 0,
        'health_score': round(100 - (negative / total) * 100, 1) if total > 0 else 100
    }


def get_sentiment_insights(df: pd.DataFrame) -> Dict:
    """
    Get detailed sentiment insights.
    
    Args:
        df: DataFrame with sentiment column
        
    Returns:
        Dictionary with sentiment insights
    """
    if 'sentiment' not in df.columns:
        return {'error': 'No sentiment data available'}
    
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    total = len(df)
    
    # Calculate percentages
    percentages = {
        sentiment: round((count / total) * 100, 1)
        for sentiment, count in sentiment_counts.items()
    }
    
    return {
        'distribution': sentiment_counts,
        'percentages': percentages,
        'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get),
        'needs_attention': percentages.get('negative', 0) > 30
    }


def get_top_issues_insights(cluster_info: Dict, top_n: int = 5) -> List[Dict]:
    """
    Get insights about top complaint issues.
    
    Args:
        cluster_info: Dictionary with cluster information
        top_n: Number of top issues to return
        
    Returns:
        List of issue insights
    """
    if not cluster_info:
        return []
    
    issues = []
    for cluster_id, info in cluster_info.items():
        issues.append({
            'id': cluster_id,
            'topic': info.get('label', f'Topic {cluster_id}'),
            'complaint_count': info.get('count', 0),
            'keywords': info.get('keywords', []),
            'example_complaints': info.get('sample_texts', [])[:2],
            'business_impact': categorize_issue_impact(info.get('keywords', []))
        })
    
    # Sort by count
    issues.sort(key=lambda x: x['complaint_count'], reverse=True)
    
    return issues[:top_n]


def categorize_issue_impact(keywords: List[str]) -> str:
    """
    Categorize the business impact based on keywords.
    
    Args:
        keywords: List of keywords from cluster
        
    Returns:
        Impact category string
    """
    high_impact_keywords = {
        'crash', 'bug', 'error', 'broken', 'fix', 'issue', 'problem',
        'login', 'account', 'ban', 'banned', 'block', 'blocked',
        'money', 'pay', 'payment', 'refund', 'scam', 'fraud'
    }
    
    medium_impact_keywords = {
        'slow', 'lag', 'loading', 'download', 'update', 'notification',
        'feature', 'missing', 'need', 'want', 'add', 'please'
    }
    
    keywords_lower = [k.lower() for k in keywords]
    
    if any(k in high_impact_keywords for k in keywords_lower):
        return "🔴 Critical - Affects core functionality"
    elif any(k in medium_impact_keywords for k in keywords_lower):
        return "🟡 Medium - Affects user experience"
    else:
        return "🟢 Low - General feedback"


def get_sample_complaints(df: pd.DataFrame, n_samples: int = 10) -> List[Dict]:
    """
    Get sample complaints for display.
    
    Args:
        df: DataFrame with reviews
        n_samples: Number of samples to return
        
    Returns:
        List of sample complaint dictionaries
    """
    if 'sentiment' not in df.columns:
        negative_df = df
    else:
        negative_df = df[df['sentiment'] == 'negative']
    
    if len(negative_df) == 0:
        return []
    
    samples = negative_df.head(n_samples)
    
    complaints = []
    for _, row in samples.iterrows():
        complaint = {
            'content': row.get('content', row.get('cleaned_content', '')),
            'score': row.get('score', 'N/A'),
            'cluster': row.get('cluster_label', 'Uncategorized')
        }
        
        if 'userName' in row:
            complaint['user'] = row['userName']
        if 'app_id' in row:
            complaint['app'] = row['app_id']
        if 'at' in row:
            complaint['date'] = str(row['at'])[:10]
            
        complaints.append(complaint)
    
    return complaints


def generate_recommendations(df: pd.DataFrame, cluster_info: Dict) -> List[str]:
    """
    Generate actionable recommendations based on analysis.
    
    Args:
        df: DataFrame with reviews
        cluster_info: Cluster information
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Check sentiment distribution
    if 'sentiment' in df.columns:
        negative_pct = len(df[df['sentiment'] == 'negative']) / len(df) * 100
        
        if negative_pct > 40:
            recommendations.append(
                "⚠️ URGENT: Over 40% negative reviews. Immediate attention needed on core issues."
            )
        elif negative_pct > 25:
            recommendations.append(
                "📊 Elevated negative sentiment. Consider prioritizing top complaint categories."
            )
    
    # Recommendations based on top issues
    if cluster_info:
        sorted_clusters = sorted(
            cluster_info.items(), 
            key=lambda x: x[1].get('count', 0), 
            reverse=True
        )
        
        if sorted_clusters:
            top_cluster = sorted_clusters[0][1]
            top_keywords = top_cluster.get('keywords', [])
            
            if any(k in ['login', 'account', 'ban', 'banned'] for k in top_keywords):
                recommendations.append(
                    "🔐 Account/login issues are prominent. Review authentication flow and ban policies."
                )
            
            if any(k in ['slow', 'lag', 'loading', 'crash'] for k in top_keywords):
                recommendations.append(
                    "⚡ Performance complaints detected. Consider performance optimization sprint."
                )
            
            if any(k in ['scam', 'fraud', 'spam'] for k in top_keywords):
                recommendations.append(
                    "🛡️ Trust & safety concerns raised. Strengthen moderation and user protection."
                )
    
    if not recommendations:
        recommendations.append(
            "✅ No critical issues detected. Continue monitoring user feedback."
        )
    
    return recommendations


def get_app_breakdown(df: pd.DataFrame) -> Dict:
    """
    Get breakdown of reviews by app.
    
    Args:
        df: DataFrame with app_id column
        
    Returns:
        Dictionary with per-app statistics
    """
    if 'app_id' not in df.columns:
        return {}
    
    breakdown = {}
    for app_id in df['app_id'].unique():
        app_df = df[df['app_id'] == app_id]
        
        sentiment_dist = {}
        if 'sentiment' in app_df.columns:
            sentiment_dist = app_df['sentiment'].value_counts().to_dict()
        
        breakdown[app_id] = {
            'total_reviews': len(app_df),
            'sentiment_distribution': sentiment_dist,
            'negative_percentage': round(
                sentiment_dist.get('negative', 0) / len(app_df) * 100, 1
            ) if len(app_df) > 0 else 0
        }
    
    return breakdown


def format_insights_report(insights: Dict) -> str:
    """
    Format insights as a readable text report.
    
    Args:
        insights: Dictionary of insights
        
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("📊 AI PRODUCT FEEDBACK ANALYSIS REPORT")
    report.append("=" * 60)
    
    # Summary
    summary = insights.get('summary', {})
    report.append(f"\n📈 OVERVIEW")
    report.append(f"   Total Reviews Analyzed: {summary.get('total_reviews', 0)}")
    report.append(f"   Positive: {summary.get('positive_count', 0)} ({summary.get('positive_percentage', 0)}%)")
    report.append(f"   Neutral: {summary.get('neutral_count', 0)}")
    report.append(f"   Negative: {summary.get('negative_count', 0)} ({summary.get('negative_percentage', 0)}%)")
    report.append(f"   Product Health Score: {summary.get('health_score', 0)}%")
    
    # Top Issues
    top_issues = insights.get('top_issues', [])
    if top_issues:
        report.append(f"\n🎯 TOP COMPLAINT CATEGORIES")
        for i, issue in enumerate(top_issues[:5], 1):
            report.append(f"\n   {i}. {issue['topic']}")
            report.append(f"      Complaints: {issue['complaint_count']}")
            report.append(f"      Impact: {issue['business_impact']}")
    
    # Recommendations
    recommendations = insights.get('recommendations', [])
    if recommendations:
        report.append(f"\n💡 RECOMMENDATIONS")
        for rec in recommendations:
            report.append(f"   • {rec}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test insights module
    from data_loader import load_reviews, DEFAULT_DATA_PATH
    from preprocess import preprocess_reviews
    from clustering import cluster_reviews, get_top_issues
    
    # Load and process data
    df = load_reviews(DEFAULT_DATA_PATH)
    processed_df = preprocess_reviews(df, filter_english=True)
    clustered_df, cluster_info = cluster_reviews(processed_df, n_clusters=6)
    
    # Generate insights
    insights = generate_insights(clustered_df, cluster_info)
    
    # Print formatted report
    print(format_insights_report(insights))
