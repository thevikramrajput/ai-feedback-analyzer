"""
AI Product Feedback Analyzer - Streamlit Dashboard
Hugging Face Spaces Compatible Version with AI Summarization
Version: 3.0 - Pastel UI Redesign
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import os
from collections import Counter

# ============== COLOR PALETTE ==============
COLORS = {
    'soft_blue': '#99CDD8',
    'mint': '#DAEBE3',
    'cream': '#FDE8D3',
    'peach': '#F3C3B2',
    'sage': '#CFDBC4',
    'text': '#657166',
    'text_light': '#7d8a7e',
    'white': '#FFFFFF',
    'shadow': 'rgba(101, 113, 102, 0.1)'
}

# Page configuration
st.set_page_config(
    page_title="AI Product Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Pastel Color Palette
st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['mint']} 0%, {COLORS['soft_blue']} 100%);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {COLORS['sage']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS['text']} !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {COLORS['text']} !important;
    }}
    
    /* Main Header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS['cream']} 0%, {COLORS['peach']} 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px {COLORS['shadow']};
        border: 1px solid rgba(255,255,255,0.3);
    }}
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        color: {COLORS['text']} !important;
        font-weight: 700;
    }}
    .main-header p {{
        margin: 0.5rem 0 0 0;
        color: {COLORS['text_light']} !important;
        font-size: 1.1rem;
    }}
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text']} !important;
    }}
    
    /* Text */
    p, span, div, label {{
        color: {COLORS['text']};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['peach']} 0%, {COLORS['cream']} 100%) !important;
        color: {COLORS['text']} !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px {COLORS['shadow']} !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {COLORS['shadow']} !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['soft_blue']} 0%, {COLORS['sage']} 100%) !important;
    }}
    
    /* Select boxes */
    .stSelectbox > div > div {{
        background: {COLORS['white']} !important;
        border: 2px solid {COLORS['sage']} !important;
        border-radius: 10px !important;
        color: {COLORS['text']} !important;
    }}
    
    /* File uploader */
    .stFileUploader {{
        background: {COLORS['white']} !important;
        border-radius: 12px !important;
        border: 2px dashed {COLORS['sage']} !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: {COLORS['cream']} !important;
        border-radius: 10px !important;
        color: {COLORS['text']} !important;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    [data-testid="stDataFrame"] {{
        background: {COLORS['white']} !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px {COLORS['shadow']} !important;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
    }}
    
    /* Success/Info messages */
    .stSuccess {{
        background: {COLORS['sage']} !important;
        color: {COLORS['text']} !important;
    }}
    .stInfo {{
        background: {COLORS['soft_blue']} !important;
        color: {COLORS['text']} !important;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-color: {COLORS['peach']} !important;
    }}
    
    /* Hide default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Cards */
    .kpi-card {{
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px {COLORS['shadow']};
        border: 1px solid rgba(255,255,255,0.5);
        transition: transform 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
    }}
    .kpi-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['text']};
    }}
    .kpi-label {{
        font-size: 0.95rem;
        color: {COLORS['text_light']};
        margin-top: 0.5rem;
    }}
    
    /* Summary cards */
    .summary-card {{
        background: {COLORS['white']};
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid {COLORS['soft_blue']};
        box-shadow: 0 4px 15px {COLORS['shadow']};
        margin: 1rem 0;
    }}
    .summary-card h4 {{
        color: {COLORS['text']} !important;
        margin: 0 0 0.5rem 0;
    }}
    .summary-card p {{
        color: {COLORS['text_light']};
        margin: 0;
        line-height: 1.6;
    }}
    
    /* Category cards */
    .category-card {{
        background: {COLORS['cream']};
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['peach']};
        margin: 0.5rem 0;
    }}
    
    /* Quote cards */
    .quote-card {{
        background: {COLORS['white']};
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid {COLORS['sage']};
        margin: 0.5rem 0;
        font-style: italic;
        color: {COLORS['text_light']};
    }}
    
    /* Action items */
    .action-item {{
        background: {COLORS['white']};
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['soft_blue']};
        margin: 0.5rem 0;
        color: {COLORS['text']};
    }}
    
    /* Welcome card */
    .welcome-card {{
        background: linear-gradient(135deg, {COLORS['cream']} 0%, {COLORS['peach']} 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px {COLORS['shadow']};
    }}
    .welcome-card h2 {{
        color: {COLORS['text']} !important;
        margin: 0;
    }}
    .welcome-card p {{
        color: {COLORS['text_light']};
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
    }}
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
        
        input_text = f"summarize: {text[:1500]}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
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
    
    # 1. OVERALL SUMMARY
    all_reviews = df['content'].astype(str).tolist()[:50]
    combined_all = " | ".join([r[:150] for r in all_reviews if len(r) > 20])
    
    if summarizer and combined_all:
        report['overall_summary'] = summarize_text(
            f"User reviews summary: {combined_all}", 
            summarizer, max_len=200
        ) or "Could not generate overall summary."
    
    # 2. NEGATIVE REVIEWS SUMMARY
    negative_df = df[df['sentiment'] == 'negative'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(negative_df) > 0:
        neg_reviews = negative_df['content'].astype(str).tolist()[:40]
        combined_neg = " | ".join([r[:150] for r in neg_reviews if len(r) > 20])
        
        if summarizer and combined_neg:
            report['negative_summary'] = summarize_text(
                f"Users are complaining that: {combined_neg}",
                summarizer, max_len=200
            ) or "Could not generate negative summary."
        
        for review in neg_reviews[:5]:
            if len(review) > 30:
                report['top_complaints'].append(review[:200])
    
    # 3. POSITIVE REVIEWS SUMMARY
    positive_df = df[df['sentiment'] == 'positive'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(positive_df) > 0:
        pos_reviews = positive_df['content'].astype(str).tolist()[:30]
        combined_pos = " | ".join([r[:150] for r in pos_reviews if len(r) > 20])
        
        if summarizer and combined_pos:
            report['positive_summary'] = summarize_text(
                f"Users love that: {combined_pos}",
                summarizer, max_len=150
            ) or "Could not generate positive summary."
        
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
    
    # 5. ACTION ITEMS
    if report['negative_summary']:
        report['action_items'].append(f"🔴 Address complaints: {report['negative_summary'][:200]}")
    
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
    """Simple keyword-based clustering."""
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
                'original_texts': clusters[i]['original']
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
        <div class="kpi-card" style="background: linear-gradient(135deg, {COLORS['soft_blue']} 0%, {COLORS['mint']} 100%);">
            <div class="kpi-value">{summary.get('total_reviews', 0):,}</div>
            <div class="kpi-label">📝 Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, {COLORS['sage']} 0%, {COLORS['mint']} 100%);">
            <div class="kpi-value">{summary.get('positive_percentage', 0)}%</div>
            <div class="kpi-label">😊 Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, {COLORS['peach']} 0%, {COLORS['cream']} 100%);">
            <div class="kpi-value">{summary.get('negative_percentage', 0)}%</div>
            <div class="kpi-label">😟 Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health = summary.get('health_score', 0)
        st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, {COLORS['cream']} 0%, {COLORS['sage']} 100%);">
            <div class="kpi-value">{health}%</div>
            <div class="kpi-label">💪 Health Score</div>
        </div>
        """, unsafe_allow_html=True)


def render_sentiment_chart(summary):
    st.markdown(f"<h3 style='color: {COLORS['text']};'>📊 Sentiment Distribution</h3>", unsafe_allow_html=True)
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[summary.get('positive_count', 0), summary.get('neutral_count', 0), summary.get('negative_count', 0)],
        hole=0.4,
        marker_colors=[COLORS['sage'], COLORS['soft_blue'], COLORS['peach']],
        textinfo='label+percent',
        textfont=dict(color=COLORS['text'], size=14)
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(font=dict(color=COLORS['text'])),
        margin=dict(t=20, b=60, l=20, r=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_issues_chart(top_issues):
    st.markdown(f"<h3 style='color: {COLORS['text']};'>🎯 Top Complaint Categories</h3>", unsafe_allow_html=True)
    
    if not top_issues:
        st.info("No issues found.")
        return
    
    labels = [issue.get('label', 'Unknown')[:25] for issue in top_issues]
    counts = [issue.get('count', 0) for issue in top_issues]
    
    # Alternate colors from palette
    bar_colors = [COLORS['peach'], COLORS['soft_blue'], COLORS['sage'], COLORS['cream'], COLORS['mint'], COLORS['peach']]
    
    fig = go.Figure(data=[go.Bar(
        x=counts, y=labels, orientation='h',
        marker=dict(color=bar_colors[:len(labels)]),
        text=counts, textposition='outside',
        textfont=dict(color=COLORS['text'])
    )])
    
    fig.update_layout(
        xaxis=dict(title_font=dict(color=COLORS['text']), tickfont=dict(color=COLORS['text'])),
        yaxis=dict(autorange="reversed", tickfont=dict(color=COLORS['text'])),
        margin=dict(t=20, b=40, l=160, r=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_ai_summary(report):
    """Render the detailed AI summary with pastel styling."""
    st.markdown("---")
    st.markdown(f"<h2 style='color: {COLORS['text']};'>🤖 AI-Generated Review Summary</h2>", unsafe_allow_html=True)
    st.caption("Based on analysis of actual user reviews")
    
    # Overall Summary
    if report.get('overall_summary'):
        st.markdown(f"""
        <div class="summary-card">
            <h4>📋 Overall Summary</h4>
            <p>{report['overall_summary']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Two columns for Negative and Positive
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h4 style='color: {COLORS['text']};'>😟 What Users Are Complaining About</h4>", unsafe_allow_html=True)
        if report.get('negative_summary'):
            st.markdown(f"""
            <div class="category-card" style="border-left-color: {COLORS['peach']};">
                <p style="color: {COLORS['text']}; margin: 0;">{report['negative_summary']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if report.get('top_complaints'):
            st.markdown(f"<p style='color: {COLORS['text']}; font-weight: 600;'>💬 Sample Complaints:</p>", unsafe_allow_html=True)
            for complaint in report['top_complaints'][:3]:
                st.markdown(f"""<div class="quote-card">"{complaint}"</div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h4 style='color: {COLORS['text']};'>😊 What Users Love About the App</h4>", unsafe_allow_html=True)
        if report.get('positive_summary'):
            st.markdown(f"""
            <div class="category-card" style="border-left-color: {COLORS['sage']};">
                <p style="color: {COLORS['text']}; margin: 0;">{report['positive_summary']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if report.get('top_praises'):
            st.markdown(f"<p style='color: {COLORS['text']}; font-weight: 600;'>💬 Sample Praises:</p>", unsafe_allow_html=True)
            for praise in report['top_praises'][:3]:
                st.markdown(f"""<div class="quote-card" style="border-left-color: {COLORS['sage']};">"{praise}"</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Category-wise summaries
    st.markdown(f"<h4 style='color: {COLORS['text']};'>📂 Category-Wise Summary</h4>", unsafe_allow_html=True)
    
    categories = [
        ('🔐 Login/Account', report.get('login_summary'), COLORS['peach']),
        ('🐛 Bugs & Performance', report.get('bugs_summary'), COLORS['soft_blue']),
        ('✨ Feature Requests', report.get('features_summary'), COLORS['sage']),
        ('💬 Messaging Issues', report.get('messaging_summary'), COLORS['cream']),
    ]
    
    cols = st.columns(2)
    for i, (title, summary, color) in enumerate(categories):
        if summary:
            with cols[i % 2]:
                st.markdown(f"""
                <div class="category-card" style="border-left-color: {color}; background: {COLORS['white']};">
                    <h5 style="color: {COLORS['text']}; margin: 0 0 0.5rem 0;">{title}</h5>
                    <p style="color: {COLORS['text_light']}; margin: 0; font-size: 0.95rem;">{summary}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Action Items
    if report.get('action_items'):
        st.markdown("---")
        st.markdown(f"<h4 style='color: {COLORS['text']};'>🎯 Key Action Items</h4>", unsafe_allow_html=True)
        for item in report['action_items'][:5]:
            st.markdown(f"""<div class="action-item">{item}</div>""", unsafe_allow_html=True)


def render_category_details(top_issues):
    """Render expandable sections for each category with ALL reviews."""
    st.markdown(f"<h3 style='color: {COLORS['text']};'>📂 Reviews by Category</h3>", unsafe_allow_html=True)
    
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
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2 style='color: {COLORS['text']};'>📁 Data Source</h2>", unsafe_allow_html=True)
        
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
                st.session_state.analyzed = False
                st.success(f"✅ {len(st.session_state.df):,} reviews loaded")
        else:
            if st.session_state.df is None:
                st.session_state.df = load_sample_data()
            if st.session_state.df is not None:
                st.success(f"✅ {len(st.session_state.df):,} reviews ready")
        
        st.divider()
        
        st.markdown(f"<h3 style='color: {COLORS['text']};'>🔍 Filters</h3>", unsafe_allow_html=True)
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
        
        app_filter = "All"
        if st.session_state.df is not None and 'app_id' in st.session_state.df.columns:
            apps = ["All"] + list(st.session_state.df['app_id'].dropna().unique())
            app_filter = st.selectbox("App", apps)
        
        st.divider()
        
        # Analyze Button
        if st.session_state.df is not None:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.session_state.analyzed = True
                st.rerun()
        
        st.divider()
        st.caption("v3.0 - Pastel UI")
    
    # Main content
    df = st.session_state.df
    
    if df is not None and st.session_state.analyzed:
        with st.spinner("🔍 Analyzing reviews..."):
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
                    with st.spinner("🧠 Reading all reviews and generating summary..."):
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
            st.markdown(f"<h3 style='color: {COLORS['text']};'>📋 All Reviews ({sentiment_filter})</h3>", unsafe_allow_html=True)
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
    elif df is not None and not st.session_state.analyzed:
        # Show welcome message before analysis
        st.markdown("""
        <div class="welcome-card">
            <h2>👈 Click "Start Analysis" to Begin</h2>
            <p>Select your data source and filters, then click the button in the sidebar to analyze your reviews.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Select a data source to get started!")


if __name__ == "__main__":
    main()
