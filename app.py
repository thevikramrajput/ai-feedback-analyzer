"""
AI Product Feedback Analyzer - Streamlit Dashboard
Hugging Face Spaces Compatible Version with AI Summarization
Version: 2.1 - Fixed T5 summarization
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


# ============== AI SUMMARIZATION ==============

@st.cache_resource
def load_summarizer():
    """Load the T5 summarization model (cached)."""
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        import torch
        
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return {'tokenizer': tokenizer, 'model': model}
    except Exception as e:
        st.warning(f"AI Model loading failed: {e}")
        return None


def summarize_text(text, summarizer, max_len=150):
    """Generate summary for given text using T5."""
    if not summarizer or not text or len(text.strip()) < 50:
        return None
    
    try:
        tokenizer = summarizer['tokenizer']
        model = summarizer['model']
        
        # Prepare input for T5
        input_text = f"summarize: {text[:1500]}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        outputs = model.generate(
            inputs,
            max_length=max_len,
            min_length=30,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return None


def generate_detailed_ai_summary(df, cluster_info, summarizer):
    """Generate comprehensive AI summary by actually reading the reviews."""
    
    report = {
        'overall_summary': '',
        'negative_summary': '',
        'positive_summary': '',
        'login_summary': '',
        'bugs_summary': '',
        'features_summary': '',
        'messaging_summary': '',
        'top_complaints': [],
        'top_praises': [],
        'action_items': []
    }
    
    # 1. OVERALL SUMMARY - Combine sample reviews
    all_reviews = df['content'].astype(str).tolist()[:50]
    combined_all = " | ".join([r[:150] for r in all_reviews if len(r) > 20])
    
    if summarizer and combined_all:
        report['overall_summary'] = summarize_text(
            f"User reviews summary: {combined_all}", 
            summarizer, max_len=200
        ) or "Could not generate overall summary."
    
    # 2. NEGATIVE REVIEWS SUMMARY - What users are complaining about
    negative_df = df[df['sentiment'] == 'negative'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(negative_df) > 0:
        neg_reviews = negative_df['content'].astype(str).tolist()[:40]
        combined_neg = " | ".join([r[:150] for r in neg_reviews if len(r) > 20])
        
        if summarizer and combined_neg:
            report['negative_summary'] = summarize_text(
                f"Users are complaining that: {combined_neg}",
                summarizer, max_len=200
            ) or "Could not generate negative summary."
        
        # Extract actual complaints as quotes
        for review in neg_reviews[:5]:
            if len(review) > 30:
                report['top_complaints'].append(review[:200])
    
    # 3. POSITIVE REVIEWS SUMMARY - What users love
    positive_df = df[df['sentiment'] == 'positive'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(positive_df) > 0:
        pos_reviews = positive_df['content'].astype(str).tolist()[:30]
        combined_pos = " | ".join([r[:150] for r in pos_reviews if len(r) > 20])
        
        if summarizer and combined_pos:
            report['positive_summary'] = summarize_text(
                f"Users love that: {combined_pos}",
                summarizer, max_len=150
            ) or "Could not generate positive summary."
        
        # Extract actual praises
        for review in pos_reviews[:3]:
            if len(review) > 30:
                report['top_praises'].append(review[:200])
    
    # 4. CATEGORY-SPECIFIC SUMMARIES
    category_labels = {
        'Login/Account Issues': 'login_summary',
        'Performance/Bugs': 'bugs_summary',
        'Feature Requests': 'features_summary',
        'Notifications/Messaging': 'messaging_summary'
    }
    
    for cluster_id, info in cluster_info.items():
        label = info.get('label', '')
        if label in category_labels:
            texts = info.get('all_texts', info.get('sample_texts', []))
            if texts:
                # Get ORIGINAL reviews (not cleaned) for better summary
                combined = " | ".join([str(t)[:150] for t in texts[:25] if len(str(t)) > 20])
                
                if summarizer and combined:
                    key = category_labels[label]
                    prompt = {
                        'login_summary': f"Users report login and account problems: {combined}",
                        'bugs_summary': f"Users report bugs and performance issues: {combined}",
                        'features_summary': f"Users are requesting these features: {combined}",
                        'messaging_summary': f"Users complain about messaging and notifications: {combined}"
                    }
                    report[key] = summarize_text(
                        prompt.get(key, combined),
                        summarizer, max_len=150
                    ) or f"Found {len(texts)} reviews about {label}"
    
    # 5. ACTION ITEMS based on actual review content
    if report['negative_summary']:
        report['action_items'].append(f"🔴 Address user complaints: {report['negative_summary'][:200]}")
    
    for label, key in category_labels.items():
        if report.get(key):
            icon = {'login_summary': '🔐', 'bugs_summary': '🐛', 'features_summary': '✨', 'messaging_summary': '💬'}.get(key, '📌')
            report['action_items'].append(f"{icon} {label}: {report[key][:150]}...")
    
    return report


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
                 'this', 'that', 'i', 'me', 'my', 'you', 'your', 'we', 'they', 'app',
                 'not', 'just', 'very', 'much', 'really', 'can', 'get', 'use', 'one'}
    
    word_counts = Counter()
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
        words = [w for w in words if w not in stopwords]
        word_counts.update(words)
    return [word for word, _ in word_counts.most_common(top_n)]


def simple_cluster(texts, original_reviews, n_clusters=6):
    """Simple keyword-based clustering. Returns both cleaned and original texts."""
    clusters = {i: {'cleaned': [], 'original': []} for i in range(n_clusters)}
    
    issue_keywords = [
        ['login', 'account', 'password', 'sign', 'ban', 'banned', 'block', 'access'],
        ['slow', 'lag', 'loading', 'crash', 'freeze', 'bug', 'error', 'stuck'],
        ['update', 'version', 'download', 'install', 'work', 'open'],
        ['notification', 'message', 'chat', 'call', 'video', 'send', 'receive'],
        ['ads', 'advertisement', 'spam', 'scam', 'fake', 'privacy'],
        ['feature', 'need', 'want', 'please', 'add', 'missing', 'wish']
    ]
    
    for i, text in enumerate(texts):
        text_lower = str(text).lower()
        orig = original_reviews[i] if i < len(original_reviews) else text
        assigned = False
        for cluster_id, keywords in enumerate(issue_keywords):
            if any(kw in text_lower for kw in keywords):
                clusters[cluster_id]['cleaned'].append(text)
                clusters[cluster_id]['original'].append(orig)
                assigned = True
                break
        if not assigned:
            clusters[n_clusters - 1]['cleaned'].append(text)
            clusters[n_clusters - 1]['original'].append(orig)
    
    cluster_info = {}
    labels = ['Login/Account Issues', 'Performance/Bugs', 'Updates/Installation',
              'Notifications/Messaging', 'Ads/Spam/Privacy', 'Feature Requests']
    
    for i in range(n_clusters):
        if clusters[i]['cleaned']:
            cluster_info[i] = {
                'label': labels[i] if i < len(labels) else f'Topic {i}',
                'keywords': extract_keywords(clusters[i]['cleaned']),
                'count': len(clusters[i]['cleaned']),
                'sample_texts': clusters[i]['cleaned'][:10],
                'all_texts': clusters[i]['cleaned'],
                'original_texts': clusters[i]['original']  # Keep original for summary
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
    
    # Pass both cleaned and original texts
    cluster_info = simple_cluster(
        negative_df['cleaned_content'].tolist(),
        negative_df['content'].tolist()
    ) if len(negative_df) > 0 else {}
    
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
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('total_reviews', 0):,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">📝 Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('positive_percentage', 0)}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('negative_percentage', 0)}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">😟 Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health = summary.get('health_score', 0)
        color = "#11998e" if health >= 70 else "#f5576c" if health < 50 else "#ffa726"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, #ffffff22 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
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


def render_ai_summary(report):
    """Render the detailed AI summary showing what users actually said."""
    st.markdown("---")
    st.header("🤖 AI-Generated Review Summary")
    st.caption("Based on analysis of actual user reviews")
    
    # Overall Summary
    if report.get('overall_summary'):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid #3b82f6; margin: 1rem 0;">
            <h4 style="color: #3b82f6; margin: 0;">📋 Overall Summary</h4>
            <p style="color: #e2e8f0; margin: 0.5rem 0 0 0; font-size: 1.1rem; line-height: 1.6;">
                {report['overall_summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Two columns for Negative and Positive
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 😟 What Users Are Complaining About")
        if report.get('negative_summary'):
            st.markdown(f"""
            <div style="background: #1e293b; padding: 1rem; border-radius: 10px; border-left: 4px solid #ef4444;">
                <p style="color: #fca5a5; margin: 0; font-size: 1rem; line-height: 1.5;">
                    {report['negative_summary']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show actual complaint quotes
        if report.get('top_complaints'):
            st.markdown("**💬 Sample User Complaints:**")
            for i, complaint in enumerate(report['top_complaints'][:3], 1):
                st.markdown(f"""
                <div style="background: #292524; padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;
                            border-left: 3px solid #f87171; font-style: italic;">
                    <span style="color: #d6d3d1;">"{complaint}"</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 😊 What Users Love About the App")
        if report.get('positive_summary'):
            st.markdown(f"""
            <div style="background: #1e293b; padding: 1rem; border-radius: 10px; border-left: 4px solid #22c55e;">
                <p style="color: #86efac; margin: 0; font-size: 1rem; line-height: 1.5;">
                    {report['positive_summary']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show actual praise quotes
        if report.get('top_praises'):
            st.markdown("**💬 Sample User Praises:**")
            for i, praise in enumerate(report['top_praises'][:3], 1):
                st.markdown(f"""
                <div style="background: #1a2e1a; padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;
                            border-left: 3px solid #4ade80; font-style: italic;">
                    <span style="color: #d6d3d1;">"{praise}"</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Category-wise summaries
    st.markdown("### 📂 Category-Wise Summary (What Users Said)")
    
    categories = [
        ('🔐 Login/Account Issues', report.get('login_summary'), '#ef4444'),
        ('🐛 Bugs & Performance', report.get('bugs_summary'), '#f97316'),
        ('✨ Feature Requests', report.get('features_summary'), '#a855f7'),
        ('💬 Messaging Issues', report.get('messaging_summary'), '#06b6d4'),
    ]
    
    cols = st.columns(2)
    for i, (title, summary, color) in enumerate(categories):
        if summary:
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 10px; 
                            border-left: 4px solid {color}; margin: 0.5rem 0; min-height: 120px;">
                    <h5 style="color: {color}; margin: 0 0 0.5rem 0;">{title}</h5>
                    <p style="color: #cbd5e1; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                        {summary}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Action Items
    if report.get('action_items'):
        st.markdown("---")
        st.markdown("### 🎯 Key Action Items (Based on User Feedback)")
        for item in report['action_items'][:5]:
            st.markdown(f"""
            <div style="background: #1e293b; padding: 0.8rem 1rem; border-radius: 8px; margin: 0.3rem 0;
                        border-left: 4px solid #3b82f6;">
                <span style="color: #e2e8f0;">{item}</span>
            </div>
            """, unsafe_allow_html=True)


def render_category_details(top_issues):
    """Render expandable sections for each category with ALL reviews."""
    st.subheader("📂 Reviews by Category")
    
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
        all_texts = issue.get('original_texts', issue.get('all_texts', []))
        
        with st.expander(f"{icon} {label} ({count} reviews)", expanded=False):
            if keywords:
                st.markdown(f"**🏷️ Keywords:** {', '.join(keywords)}")
            
            st.markdown("---")
            
            if all_texts:
                # Show ALL reviews with pagination
                reviews_df = pd.DataFrame({
                    '#': range(1, len(all_texts) + 1),
                    'Review': [str(text)[:500] for text in all_texts]
                })
                st.dataframe(reviews_df, use_container_width=True, hide_index=True, height=400)
                st.success(f"Showing all {len(all_texts)} reviews in this category")


# ============== MAIN APP ==============

def main():
    render_header()
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = None
    
    # Sidebar
    with st.sidebar:
        st.title("📁 Data Source")
        
        data_source = st.radio(
            "Choose data source:",
            ["📊 Sample Dataset", "📤 Upload CSV"],
            index=0
        )
        
        if data_source == "📤 Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.ai_report = None
                st.success(f"✅ {len(st.session_state.df):,} reviews")
        else:
            if st.session_state.df is None:
                st.session_state.df = load_sample_data()
            if st.session_state.df is not None:
                st.success(f"✅ {len(st.session_state.df):,} reviews")
        
        st.divider()
        
        st.subheader("🔍 Filters")
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
        
        app_filter = "All"
        if st.session_state.df is not None and 'app_id' in st.session_state.df.columns:
            apps = ["All"] + list(st.session_state.df['app_id'].dropna().unique())
            app_filter = st.selectbox("App", apps)
        
        st.divider()
        st.caption("v2.0 - Powered by DistilBART")
    
    # Main content
    df = st.session_state.df
    
    if df is not None:
        with st.spinner("🔍 Analyzing..."):
            results = process_data(df, sentiment_filter, app_filter)
        
        if results:
            render_kpi_cards(results['summary'])
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                render_sentiment_chart(results['summary'])
            with col2:
                render_issues_chart(results['top_issues'])
            
            # AI Summary Button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🤖 Generate Detailed AI Summary", type="primary", use_container_width=True):
                    with st.spinner("🧠 Reading all reviews and generating summary... Please wait..."):
                        summarizer = load_summarizer()
                        report = generate_detailed_ai_summary(
                            results['processed_df'],
                            results['cluster_info'],
                            summarizer
                        )
                        st.session_state.ai_report = report
            
            # Display AI Summary
            if st.session_state.ai_report:
                render_ai_summary(st.session_state.ai_report)
            
            st.divider()
            render_category_details(results['top_issues'])
            
            st.divider()
            
            # All Reviews section
            st.subheader(f"📋 All Reviews ({sentiment_filter})")
            if 'processed_df' in results and len(results['processed_df']) > 0:
                display_df = results['processed_df'].copy()
                if sentiment_filter != "All":
                    display_df = display_df[display_df['sentiment'] == sentiment_filter.lower()]
                
                if len(display_df) > 0:
                    all_reviews_df = pd.DataFrame({
                        '#': range(1, len(display_df) + 1),
                        'Review': display_df['content'].astype(str).tolist(),
                        'Score': display_df['score'].tolist() if 'score' in display_df.columns else ['N/A'] * len(display_df),
                        'Sentiment': display_df['sentiment'].str.title().tolist() if 'sentiment' in display_df.columns else ['N/A'] * len(display_df)
                    })
                    st.dataframe(all_reviews_df, use_container_width=True, hide_index=True, height=500)
                    st.success(f"Showing all {len(display_df)} reviews")
                else:
                    st.info("No reviews match the current filter.")
    else:
        st.info("👈 Select a data source to get started!")


if __name__ == "__main__":
    main()
