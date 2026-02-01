"""
AI Product Feedback Analyzer - Streamlit Dashboard
Hugging Face Spaces Compatible Version with AI Summarization
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
    .ai-summary-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #334155;
        margin: 1rem 0;
    }
    .summary-section {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# ============== AI SUMMARIZATION ==============

@st.cache_resource
def load_summarizer():
    """Load the summarization model (cached)."""
    try:
        from transformers import pipeline
        # Using a smaller, faster model for HF Spaces
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        return summarizer
    except Exception as e:
        return None


def generate_category_summary(texts, category_name, summarizer):
    """Generate summary for a specific category of reviews."""
    if not texts or not summarizer:
        return None
    
    try:
        # Combine texts (limit to avoid token limits)
        combined_text = " ".join([str(t)[:200] for t in texts[:30]])
        if len(combined_text) < 50:
            return None
        
        # Truncate to model limits
        combined_text = combined_text[:1024]
        
        summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return None


def generate_complete_ai_summary(df, cluster_info, summarizer):
    """Generate a complete AI-powered summary of all reviews."""
    summary_report = {
        'executive_summary': '',
        'areas_to_improve': [],
        'feature_requests': '',
        'critical_bugs': '',
        'login_issues': '',
        'messaging_issues': '',
        'positive_feedback': '',
        'recommendations': []
    }
    
    # Generate category-specific summaries
    category_mapping = {
        'Login/Account Issues': 'login_issues',
        'Performance/Bugs': 'critical_bugs',
        'Feature Requests': 'feature_requests',
        'Notifications/Messaging': 'messaging_issues'
    }
    
    for cluster_id, info in cluster_info.items():
        label = info.get('label', '')
        texts = info.get('all_texts', info.get('sample_texts', []))
        
        if label in category_mapping and texts:
            key = category_mapping[label]
            if summarizer:
                ai_summary = generate_category_summary(texts, label, summarizer)
                if ai_summary:
                    summary_report[key] = ai_summary
            
            # Fallback: Generate rule-based summary
            if not summary_report[key]:
                keywords = info.get('keywords', [])
                count = info.get('count', 0)
                summary_report[key] = f"Found {count} reviews mentioning: {', '.join(keywords[:5])}"
    
    # Generate positive feedback summary
    positive_df = df[df['sentiment'] == 'positive'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(positive_df) > 0 and 'cleaned_content' in positive_df.columns:
        pos_texts = positive_df['cleaned_content'].tolist()[:20]
        if summarizer:
            pos_summary = generate_category_summary(pos_texts, 'Positive', summarizer)
            if pos_summary:
                summary_report['positive_feedback'] = pos_summary
        
        if not summary_report['positive_feedback']:
            pos_keywords = extract_keywords(pos_texts, top_n=8)
            summary_report['positive_feedback'] = f"Users appreciate: {', '.join(pos_keywords)}"
    
    # Executive Summary
    total = len(df)
    neg_count = len(df[df['sentiment'] == 'negative']) if 'sentiment' in df.columns else 0
    pos_count = len(df[df['sentiment'] == 'positive']) if 'sentiment' in df.columns else 0
    
    summary_report['executive_summary'] = f"""
    Analyzed {total:,} user reviews. {round(pos_count/total*100, 1)}% positive, {round(neg_count/total*100, 1)}% negative.
    Top complaint categories identified and summarized below.
    """
    
    # Areas to improve based on clusters
    sorted_clusters = sorted(cluster_info.values(), key=lambda x: x.get('count', 0), reverse=True)
    for cluster in sorted_clusters[:4]:
        count = cluster.get('count', 0)
        label = cluster.get('label', 'Unknown')
        keywords = cluster.get('keywords', [])
        summary_report['areas_to_improve'].append({
            'area': label,
            'count': count,
            'keywords': keywords
        })
    
    # Recommendations
    summary_report['recommendations'] = [
        "🔐 **Account & Login**: Simplify authentication flow, reduce ban false-positives" if summary_report.get('login_issues') else None,
        "⚡ **Performance**: Optimize app speed, fix crash issues, reduce memory usage" if summary_report.get('critical_bugs') else None,
        "✨ **Features**: Prioritize most-requested features based on user feedback" if summary_report.get('feature_requests') else None,
        "💬 **Messaging**: Improve notification reliability and message sync" if summary_report.get('messaging_issues') else None,
    ]
    summary_report['recommendations'] = [r for r in summary_report['recommendations'] if r]
    
    return summary_report


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
                 'this', 'that', 'i', 'me', 'my', 'you', 'your', 'we', 'they', 'app', 'good', 'bad',
                 'not', 'just', 'very', 'much', 'really', 'can', 'get', 'use', 'using', 'one'}
    
    word_counts = Counter()
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        words = [w for w in words if w not in stopwords]
        word_counts.update(words)
    return [word for word, _ in word_counts.most_common(top_n)]


def simple_cluster(texts, n_clusters=6):
    """Simple keyword-based clustering."""
    clusters = {i: [] for i in range(n_clusters)}
    
    issue_keywords = [
        ['login', 'account', 'password', 'sign', 'ban', 'banned', 'block', 'access'],
        ['slow', 'lag', 'loading', 'crash', 'freeze', 'bug', 'error', 'stuck'],
        ['update', 'version', 'download', 'install', 'work', 'open'],
        ['notification', 'message', 'chat', 'call', 'video', 'send', 'receive'],
        ['ads', 'advertisement', 'spam', 'scam', 'fake', 'privacy'],
        ['feature', 'need', 'want', 'please', 'add', 'missing', 'wish']
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
    
    cluster_info = {}
    labels = ['Login/Account Issues', 'Performance/Bugs', 'Updates/Installation',
              'Notifications/Messaging', 'Ads/Spam/Privacy', 'Feature Requests']
    
    for i in range(n_clusters):
        if clusters[i]:
            cluster_info[i] = {
                'label': labels[i] if i < len(labels) else f'Topic {i}',
                'keywords': extract_keywords(clusters[i]),
                'count': len(clusters[i]),
                'sample_texts': clusters[i][:10],
                'all_texts': clusters[i]
            }
    
    return cluster_info


def process_data(df, sentiment_filter="All", app_filter="All"):
    """Full processing pipeline with filters."""
    df = df.copy()
    
    if 'content' not in df.columns:
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower() or 'content' in col.lower()]
        if text_cols:
            df['content'] = df[text_cols[0]]
        else:
            return None
    
    df = df.dropna(subset=['content'])
    df = df[df['content'].astype(str).str.strip() != ''].copy()
    df['cleaned_content'] = df['content'].apply(clean_text)
    df = df[df['cleaned_content'].str.strip() != ''].copy()
    
    if 'score' in df.columns:
        df['sentiment'] = df['score'].apply(score_to_sentiment)
    else:
        df['sentiment'] = 'neutral'
    
    if app_filter != "All" and 'app_id' in df.columns:
        df = df[df['app_id'] == app_filter].copy()
    
    total = len(df)
    if total == 0:
        return None
    
    positive_count = len(df[df['sentiment'] == 'positive'])
    negative_count = len(df[df['sentiment'] == 'negative'])
    neutral_count = len(df[df['sentiment'] == 'neutral'])
    
    display_df = df.copy()
    if sentiment_filter != "All":
        display_df = df[df['sentiment'] == sentiment_filter.lower()].copy()
    
    negative_df = df[df['sentiment'] == 'negative']
    cluster_info = simple_cluster(negative_df['cleaned_content'].tolist()) if len(negative_df) > 0 else {}
    
    summary = {
        'total_reviews': total,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'positive_percentage': round(positive_count / total * 100, 1),
        'negative_percentage': round(negative_count / total * 100, 1),
        'health_score': round(100 - (negative_count / total * 100), 1)
    }
    
    top_issues = sorted(cluster_info.values(), key=lambda x: x['count'], reverse=True)[:6]
    
    sample_df = display_df if sentiment_filter != "All" else negative_df
    sample_complaints = []
    for _, row in sample_df.head(15).iterrows():
        sample_complaints.append({
            'Review': str(row.get('content', ''))[:200],
            'Score': row.get('score', 'N/A'),
            'Sentiment': row.get('sentiment', 'N/A').title()
        })
    
    return {
        'summary': summary,
        'top_issues': top_issues,
        'sample_complaints': sample_complaints,
        'cluster_info': cluster_info,
        'filtered_count': len(display_df),
        'processed_df': df
    }


def load_sample_data():
    """Load sample data from various paths."""
    data_paths = [
        'data/Training_Data.csv',
        './data/Training_Data.csv',
        os.path.join(os.path.dirname(__file__), 'data', 'Training_Data.csv'),
    ]
    for path in data_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                continue
    return None


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


def render_ai_summary(summary_report):
    """Render the complete AI-generated summary."""
    st.markdown("---")
    st.subheader("🤖 AI-Generated Comprehensive Summary")
    
    # Executive Summary
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                padding: 1.5rem; border-radius: 12px; border: 1px solid #3b82f6; margin-bottom: 1rem;">
        <h4 style="color: #3b82f6; margin-top: 0;">📋 Executive Summary</h4>
        <p style="color: #e2e8f0;">{summary_report.get('executive_summary', 'No summary available.')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["� Areas to Improve", "💬 Category Summaries", "✅ Positive Feedback", "💡 Recommendations"])
    
    with tab1:
        areas = summary_report.get('areas_to_improve', [])
        if areas:
            for area in areas:
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid #f5576c;">
                    <strong style="color: #f5576c;">{area['area']}</strong>
                    <span style="color: #94a3b8; float: right;">{area['count']} complaints</span>
                    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        Keywords: {', '.join(area['keywords'][:5])}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific areas to improve identified.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Login Issues
            if summary_report.get('login_issues'):
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid #ef4444;">
                    <h5 style="color: #ef4444; margin: 0;">🔐 Login/Account Issues</h5>
                    <p style="color: #e2e8f0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        {summary_report['login_issues']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Bugs
            if summary_report.get('critical_bugs'):
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid #f97316;">
                    <h5 style="color: #f97316; margin: 0;">🐛 Performance/Bugs</h5>
                    <p style="color: #e2e8f0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        {summary_report['critical_bugs']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Feature Requests
            if summary_report.get('feature_requests'):
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid #a855f7;">
                    <h5 style="color: #a855f7; margin: 0;">✨ Feature Requests</h5>
                    <p style="color: #e2e8f0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        {summary_report['feature_requests']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Messaging
            if summary_report.get('messaging_issues'):
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid #06b6d4;">
                    <h5 style="color: #06b6d4; margin: 0;">💬 Messaging Issues</h5>
                    <p style="color: #e2e8f0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        {summary_report['messaging_issues']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        if summary_report.get('positive_feedback'):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
                        padding: 1.5rem; border-radius: 12px; border: 1px solid #10b981;">
                <h5 style="color: #10b981; margin: 0;">😊 What Users Love</h5>
                <p style="color: #e2e8f0; margin: 0.5rem 0 0 0;">
                    {summary_report['positive_feedback']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No positive feedback summary available.")
    
    with tab4:
        recommendations = summary_report.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid #3b82f6;">
                    <p style="color: #e2e8f0; margin: 0;">{rec}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical recommendations at this time.")


def render_category_details(top_issues):
    """Render expandable sections for each category with reviews."""
    st.subheader("📂 Reviews by Category")
    st.caption("Click on a category to view its reviews")
    
    if not top_issues:
        st.info("No categories found.")
        return
    
    icons = {
        'Login/Account Issues': '🔐',
        'Performance/Bugs': '🐛',
        'Updates/Installation': '📥',
        'Notifications/Messaging': '💬',
        'Ads/Spam/Privacy': '🚫',
        'Feature Requests': '✨'
    }
    
    for issue in top_issues:
        label = issue.get('label', 'Unknown')
        icon = icons.get(label, '📌')
        count = issue.get('count', 0)
        keywords = issue.get('keywords', [])
        all_texts = issue.get('all_texts', issue.get('sample_texts', []))
        
        with st.expander(f"{icon} {label} ({count} reviews)", expanded=False):
            if keywords:
                st.markdown(f"**🏷️ Keywords:** {', '.join(keywords)}")
            
            st.markdown("---")
            
            if all_texts:
                reviews_to_show = all_texts[:20]
                reviews_df = pd.DataFrame({
                    'Review': [str(text)[:300] + ('...' if len(str(text)) > 300 else '') for text in reviews_to_show]
                })
                st.dataframe(reviews_df, use_container_width=True, hide_index=True)
                
                if len(all_texts) > 20:
                    st.caption(f"Showing 20 of {len(all_texts)} reviews in this category")
            else:
                st.info("No reviews available")


# ============== MAIN APP ==============

def main():
    render_header()
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = None
    
    # Sidebar
    with st.sidebar:
        st.title("📁 Data Source")
        
        data_source = st.radio(
            "Choose data source:",
            ["📊 Sample Dataset", "📤 Upload CSV"],
            index=0
        )
        
        if data_source == "📤 Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV with reviews", type=['csv'])
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.ai_summary = None  # Reset summary
                st.success(f"✅ Uploaded {len(st.session_state.df):,} reviews")
        else:
            if st.session_state.df is None:
                st.session_state.df = load_sample_data()
            if st.session_state.df is not None:
                st.success(f"✅ Loaded {len(st.session_state.df):,} sample reviews")
            else:
                st.warning("Sample data not found")
        
        st.divider()
        
        st.subheader("🔍 Filters")
        
        sentiment_filter = st.selectbox(
            "Sentiment",
            ["All", "Positive", "Neutral", "Negative"]
        )
        
        app_filter = "All"
        if st.session_state.df is not None and 'app_id' in st.session_state.df.columns:
            apps = ["All"] + list(st.session_state.df['app_id'].dropna().unique())
            app_filter = st.selectbox("App", apps)
        
        st.divider()
        st.caption("AI Product Feedback Analyzer v2.0")
        st.caption("🤖 Powered by DistilBART")
    
    # Main content
    df = st.session_state.df
    
    if df is not None:
        with st.spinner("🔍 Analyzing reviews..."):
            results = process_data(df, sentiment_filter, app_filter)
        
        if results:
            if sentiment_filter != "All" or app_filter != "All":
                st.info(f"📌 Showing {results['filtered_count']:,} reviews | Filters: Sentiment={sentiment_filter}, App={app_filter}")
            
            # KPI Cards
            render_kpi_cards(results['summary'])
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                render_sentiment_chart(results['summary'])
            with col2:
                render_issues_chart(results['top_issues'])
            
            # AI Summary Generation Button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🤖 Generate AI Summary", type="primary", use_container_width=True):
                    with st.spinner("🧠 AI is analyzing all reviews... This may take a minute..."):
                        summarizer = load_summarizer()
                        ai_summary = generate_complete_ai_summary(
                            results['processed_df'], 
                            results['cluster_info'], 
                            summarizer
                        )
                        st.session_state.ai_summary = ai_summary
            
            # Display AI Summary if generated
            if st.session_state.ai_summary:
                render_ai_summary(st.session_state.ai_summary)
            
            st.divider()
            
            # Category Details
            render_category_details(results['top_issues'])
            
            st.divider()
            
            # Sample reviews table
            st.subheader(f"📋 Sample Reviews ({sentiment_filter})")
            if results['sample_complaints']:
                complaints_df = pd.DataFrame(results['sample_complaints'])
                st.dataframe(complaints_df, use_container_width=True, hide_index=True)
            else:
                st.info("No reviews match the current filters.")
        else:
            st.error("Could not process data. Ensure CSV has 'content' and 'score' columns.")
    else:
        st.info("👈 Select a data source from the sidebar to get started!")


if __name__ == "__main__":
    main()
