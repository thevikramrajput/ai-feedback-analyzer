"""
AI Product Feedback Analyzer - Streamlit Dashboard
Hugging Face Spaces Compatible Version
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import os
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="AI Product Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; color: white !important; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; color: #e2e8f0 !important; }
    h2, h3 { color: #f8fafc !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============== DATA PROCESSING FUNCTIONS ==============

def clean_text(text):
    """Clean review text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.,!?\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def score_to_sentiment(score):
    """Convert star rating to sentiment."""
    try:
        score = int(score)
        if score <= 2:
            return 'negative'
        elif score == 3:
            return 'neutral'
        return 'positive'
    except:
        return 'neutral'


def extract_keywords(texts, top_n=5):
    """Extract keywords from texts."""
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'it', 'its',
                 'this', 'that', 'i', 'me', 'my', 'you', 'your', 'we', 'they', 'app', 'good', 'bad'}
    
    word_counts = Counter()
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        words = [w for w in words if w not in stopwords]
        word_counts.update(words)
    return [word for word, _ in word_counts.most_common(top_n)]


def simple_cluster(texts, n_clusters=6):
    """Simple keyword-based clustering."""
    clusters = {i: [] for i in range(n_clusters)}
    
    # Simple keyword matching for clustering
    issue_keywords = [
        ['login', 'account', 'password', 'sign', 'ban', 'banned', 'block'],
        ['slow', 'lag', 'loading', 'crash', 'freeze', 'bug', 'error'],
        ['update', 'version', 'download', 'install', 'work'],
        ['notification', 'message', 'chat', 'call', 'video'],
        ['ads', 'advertisement', 'spam', 'scam', 'fake'],
        ['feature', 'need', 'want', 'please', 'add', 'missing']
    ]
    
    for text in texts:
        text_lower = str(text).lower()
        assigned = False
        for cluster_id, keywords in enumerate(issue_keywords):
            if any(kw in text_lower for kw in keywords):
                clusters[cluster_id].append(text)
                assigned = True
                break
        if not assigned:
            clusters[n_clusters - 1].append(text)
    
    # Generate cluster info
    cluster_info = {}
    labels = ['Login/Account Issues', 'Performance/Bugs', 'Updates/Installation',
              'Notifications/Messaging', 'Ads/Spam/Scams', 'Feature Requests']
    
    for i in range(n_clusters):
        if clusters[i]:
            cluster_info[i] = {
                'label': labels[i] if i < len(labels) else f'Topic {i}',
                'keywords': extract_keywords(clusters[i]),
                'count': len(clusters[i]),
                'sample_texts': clusters[i][:3]
            }
    
    return cluster_info


def process_data(df):
    """Full processing pipeline."""
    # Make a copy
    df = df.copy()
    
    # Filter English reviews if userLang column exists
    if 'userLang' in df.columns:
        # Handle NaN values and filter
        df['userLang'] = df['userLang'].fillna('')
        df = df[df['userLang'].str.strip().str.upper().isin(['EN', 'ENGLISH', ''])].copy()
    
    # Ensure content column exists
    if 'content' not in df.columns:
        # Try to find a text column
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower() or 'content' in col.lower()]
        if text_cols:
            df['content'] = df[text_cols[0]]
        else:
            return {
                'summary': {'total_reviews': 0, 'positive_count': 0, 'negative_count': 0, 
                           'neutral_count': 0, 'positive_percentage': 0, 'negative_percentage': 0, 'health_score': 100},
                'top_issues': [],
                'sample_complaints': [],
                'cluster_info': {}
            }
    
    # Clean and filter
    df = df.dropna(subset=['content'])
    df = df[df['content'].astype(str).str.strip() != ''].copy()
    df['cleaned_content'] = df['content'].apply(clean_text)
    df = df[df['cleaned_content'].str.strip() != ''].copy()
    
    # Add sentiment based on score
    if 'score' in df.columns:
        df['sentiment'] = df['score'].apply(score_to_sentiment)
    else:
        # Default to neutral if no score
        df['sentiment'] = 'neutral'
    
    # Calculate totals
    total = len(df)
    if total == 0:
        return {
            'summary': {'total_reviews': 0, 'positive_count': 0, 'negative_count': 0, 
                       'neutral_count': 0, 'positive_percentage': 0, 'negative_percentage': 0, 'health_score': 100},
            'top_issues': [],
            'sample_complaints': [],
            'cluster_info': {}
        }
    
    positive_count = len(df[df['sentiment'] == 'positive'])
    negative_count = len(df[df['sentiment'] == 'negative'])
    neutral_count = len(df[df['sentiment'] == 'neutral'])
    
    # Cluster negative reviews
    negative_df = df[df['sentiment'] == 'negative']
    if len(negative_df) > 0:
        cluster_info = simple_cluster(negative_df['cleaned_content'].tolist())
    else:
        cluster_info = {}
    
    # Generate summary
    summary = {
        'total_reviews': total,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'positive_percentage': round(positive_count / total * 100, 1),
        'negative_percentage': round(negative_count / total * 100, 1),
        'health_score': round(100 - (negative_count / total * 100), 1)
    }
    
    # Top issues
    top_issues = sorted(cluster_info.values(), key=lambda x: x['count'], reverse=True)[:5]
    
    # Sample complaints
    sample_complaints = []
    for _, row in negative_df.head(10).iterrows():
        sample_complaints.append({
            'content': str(row.get('content', ''))[:200],
            'score': row.get('score', 'N/A'),
            'cluster': 'Complaint'
        })
    
    return {
        'summary': summary,
        'top_issues': top_issues,
        'sample_complaints': sample_complaints,
        'cluster_info': cluster_info
    }


# ============== UI COMPONENTS ==============

def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>📊 AI Product Feedback Analyzer</h1>
            <p>Transform user reviews into actionable product insights</p>
        </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(summary):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('total_reviews', 0):,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">📝 Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('positive_percentage', 0)}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('negative_percentage', 0)}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">😟 Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health = summary.get('health_score', 0)
        color = "#11998e" if health >= 70 else "#f5576c" if health < 50 else "#ffa726"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, #ffffff22 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 2.5rem; font-weight: 700;">{health}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">💪 Health Score</div>
        </div>
        """, unsafe_allow_html=True)


def render_sentiment_chart(summary):
    st.subheader("📊 Sentiment Distribution")
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[summary.get('positive_count', 0), summary.get('neutral_count', 0), summary.get('negative_count', 0)],
        hole=0.4,
        marker_colors=['#38ef7d', '#ffa726', '#f5576c'],
        textinfo='label+percent',
        textfont=dict(color='white', size=14)
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(font=dict(color='white')),
        margin=dict(t=20, b=60, l=20, r=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_issues_chart(top_issues):
    st.subheader("🎯 Top Complaint Categories")
    
    if not top_issues:
        st.info("No issues found.")
        return
    
    labels = [issue.get('label', 'Unknown')[:25] for issue in top_issues]
    counts = [issue.get('count', 0) for issue in top_issues]
    
    fig = go.Figure(data=[go.Bar(
        x=counts, y=labels, orientation='h',
        marker=dict(color=counts, colorscale=[[0, '#ffa726'], [1, '#f5576c']]),
        text=counts, textposition='outside',
        textfont=dict(color='white')
    )])
    
    fig.update_layout(
        xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
        yaxis=dict(autorange="reversed", tickfont=dict(color='white')),
        margin=dict(t=20, b=40, l=160, r=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


# ============== MAIN APP ==============

def main():
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.title("📁 Data Upload")
        
        uploaded_file = st.file_uploader("Upload CSV with reviews", type=['csv'])
        
        use_sample = st.checkbox("Use sample data", value=False)
        
        st.divider()
        st.caption("AI Product Feedback Analyzer v1.0")
    
    # Main content
    df = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} reviews")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    elif use_sample:
        # Try to load sample data
        data_paths = [
            os.path.join(os.path.dirname(__file__), 'data', 'Training_Data.csv'),
            'data/Training_Data.csv',
            './data/Training_Data.csv'
        ]
        for data_path in data_paths:
            if os.path.exists(data_path):
                try:
                    df = pd.read_csv(data_path)
                    st.sidebar.success(f"✅ Loaded {len(df)} sample reviews")
                    break
                except:
                    continue
        
        if df is None:
            st.sidebar.warning("Sample data not found. Please upload a CSV.")
    
    if df is not None:
        # Show columns for debugging
        st.sidebar.caption(f"Columns: {', '.join(df.columns[:5])}...")
        
        # Process data
        with st.spinner("🔍 Analyzing reviews..."):
            results = process_data(df)
        
        # Display results
        render_kpi_cards(results['summary'])
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            render_sentiment_chart(results['summary'])
        with col2:
            render_issues_chart(results['top_issues'])
        
        st.divider()
        
        # Sample complaints
        st.subheader("📋 Sample Complaints")
        if results['sample_complaints']:
            complaints_df = pd.DataFrame(results['sample_complaints'])
            st.dataframe(complaints_df, use_container_width=True, hide_index=True)
        else:
            st.info("No negative reviews found.")
    else:
        st.info("👈 Upload a CSV file with 'content' and 'score' columns to analyze reviews!")
        st.markdown("""
        **Expected CSV format:**
        - `content` - Review text
        - `score` - Star rating (1-5)
        - `userLang` (optional) - Language code (e.g., 'EN')
        """)


if __name__ == "__main__":
    main()
