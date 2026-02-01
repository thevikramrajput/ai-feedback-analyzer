"""
AI Product Feedback Analyzer - Streamlit Dashboard
Version: 6.0 - Luna Blue UI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import os
from collections import Counter

# ============== LUNA COLOR PALETTE ==============
COLORS = {
    'light_cyan': '#A7EBF2',    # Light accent, highlights
    'teal': '#54ACBF',          # Primary buttons, accents
    'blue': '#26658C',          # Secondary elements
    'dark_blue': '#023859',     # Cards, content areas
    'navy': '#011C40',          # Primary background
}

# Page configuration
st.set_page_config(
    page_title="AI Product Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== LUNA BLUE CSS ==============
st.markdown(f"""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Background */
    .stApp {{
        background-color: {COLORS['navy']};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['dark_blue']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS['light_cyan']} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {COLORS['light_cyan']} !important;
        font-weight: 600;
    }}
    
    /* Hide Streamlit defaults */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Hero Header */
    .hero-card {{
        background-color: {COLORS['dark_blue']};
        padding: 2.5rem 3rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid {COLORS['blue']};
        box-shadow: 0 8px 32px rgba(1, 28, 64, 0.5);
    }}
    .hero-card h1 {{
        color: {COLORS['light_cyan']};
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .hero-card p {{
        color: {COLORS['teal']};
        font-size: 1.15rem;
        margin: 0.8rem 0 0 0;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background-color: {COLORS['dark_blue']};
        padding: 1.8rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid {COLORS['blue']};
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(1, 28, 64, 0.4);
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        border-color: {COLORS['teal']};
        box-shadow: 0 8px 32px rgba(84, 172, 191, 0.2);
    }}
    .kpi-value {{
        font-size: 2.8rem;
        font-weight: 700;
        color: {COLORS['light_cyan']};
        line-height: 1;
    }}
    .kpi-label {{
        font-size: 0.95rem;
        color: {COLORS['teal']};
        margin-top: 0.6rem;
        font-weight: 500;
    }}
    .kpi-icon {{
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }}
    
    /* Section Cards */
    .section-card {{
        background-color: {COLORS['dark_blue']};
        padding: 1.8rem;
        border-radius: 14px;
        border: 1px solid {COLORS['blue']};
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(1, 28, 64, 0.4);
    }}
    .section-card h3 {{
        color: {COLORS['light_cyan']} !important;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }}
    
    /* Summary Card */
    .summary-card {{
        background-color: {COLORS['dark_blue']};
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['teal']};
        margin: 1rem 0;
        border: 1px solid {COLORS['blue']};
    }}
    .summary-card h4 {{
        color: {COLORS['light_cyan']};
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }}
    .summary-card p {{
        color: {COLORS['teal']};
        margin: 0;
        line-height: 1.6;
    }}
    
    /* Quote Card */
    .quote-card {{
        background-color: {COLORS['navy']};
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 3px solid {COLORS['teal']};
        margin: 0.6rem 0;
        font-style: italic;
        color: {COLORS['light_cyan']};
        font-size: 0.95rem;
    }}
    
    /* Category Card */
    .category-card {{
        background-color: {COLORS['dark_blue']};
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['teal']};
        margin: 0.8rem 0;
        border: 1px solid {COLORS['blue']};
    }}
    .category-card h5 {{
        color: {COLORS['light_cyan']};
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }}
    .category-card p {{
        color: {COLORS['teal']};
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    
    /* Action Items */
    .action-item {{
        background-color: {COLORS['dark_blue']};
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['teal']};
        margin: 0.5rem 0;
        color: {COLORS['light_cyan']};
        font-weight: 500;
        border: 1px solid {COLORS['blue']};
    }}
    
    /* Welcome Card */
    .welcome-card {{
        background-color: {COLORS['dark_blue']};
        padding: 4rem 3rem;
        border-radius: 16px;
        text-align: center;
        margin: 3rem auto;
        max-width: 600px;
        border: 1px solid {COLORS['blue']};
        box-shadow: 0 8px 32px rgba(1, 28, 64, 0.5);
    }}
    .welcome-card h2 {{
        color: {COLORS['light_cyan']};
        font-size: 1.8rem;
        margin: 0;
        font-weight: 600;
    }}
    .welcome-card p {{
        color: {COLORS['teal']};
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['teal']} !important;
        color: {COLORS['navy']} !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['light_cyan']} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(84, 172, 191, 0.4) !important;
    }}
    
    /* Select boxes */
    .stSelectbox > div > div {{
        background-color: {COLORS['navy']} !important;
        border: 1px solid {COLORS['blue']} !important;
        border-radius: 10px !important;
        color: {COLORS['light_cyan']} !important;
    }}
    .stSelectbox label {{
        color: {COLORS['light_cyan']} !important;
    }}
    
    /* Radio buttons */
    .stRadio label {{
        color: {COLORS['light_cyan']} !important;
    }}
    
    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: {COLORS['navy']};
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed {COLORS['blue']};
    }}
    [data-testid="stFileUploader"] * {{
        color: {COLORS['light_cyan']} !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {COLORS['dark_blue']} !important;
        border-radius: 10px !important;
        color: {COLORS['light_cyan']} !important;
        font-weight: 500;
        border: 1px solid {COLORS['blue']} !important;
    }}
    
    /* Dataframes */
    [data-testid="stDataFrame"] {{
        background-color: {COLORS['dark_blue']};
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid {COLORS['blue']};
    }}
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {{
        background-color: {COLORS['dark_blue']} !important;
        color: {COLORS['light_cyan']} !important;
        border: 1px solid {COLORS['teal']} !important;
        border-radius: 10px !important;
    }}
    
    /* Dividers */
    hr {{
        border: none;
        height: 1px;
        background-color: {COLORS['blue']};
        margin: 2rem 0;
    }}
    
    /* All headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['light_cyan']} !important;
    }}
    
    /* Main text */
    .stApp p, .stApp span, .stApp label {{
        color: {COLORS['teal']};
    }}
    
    /* Caption */
    .stCaption {{
        color: {COLORS['blue']} !important;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {COLORS['teal']} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ============== AI SUMMARIZATION ==============

@st.cache_resource
def load_summarizer():
    """Load the T5 summarization model (cached)."""
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
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
        outputs = model.generate(inputs, max_length=max_len, min_length=30, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except:
        return None


def generate_detailed_ai_summary(df, cluster_info, summarizer):
    """Generate comprehensive AI summary."""
    report = {
        'overall_summary': '', 'negative_summary': '', 'positive_summary': '',
        'login_summary': '', 'bugs_summary': '', 'features_summary': '', 'messaging_summary': '',
        'top_complaints': [], 'top_praises': [], 'action_items': []
    }
    
    all_reviews = df['content'].astype(str).tolist()[:50]
    combined_all = " | ".join([r[:150] for r in all_reviews if len(r) > 20])
    if summarizer and combined_all:
        report['overall_summary'] = summarize_text(f"User reviews summary: {combined_all}", summarizer, max_len=200) or ""
    
    negative_df = df[df['sentiment'] == 'negative'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(negative_df) > 0:
        neg_reviews = negative_df['content'].astype(str).tolist()[:40]
        combined_neg = " | ".join([r[:150] for r in neg_reviews if len(r) > 20])
        if summarizer and combined_neg:
            report['negative_summary'] = summarize_text(f"Users are complaining: {combined_neg}", summarizer, max_len=200) or ""
        for review in neg_reviews[:5]:
            if len(review) > 30: report['top_complaints'].append(review[:200])
    
    positive_df = df[df['sentiment'] == 'positive'] if 'sentiment' in df.columns else pd.DataFrame()
    if len(positive_df) > 0:
        pos_reviews = positive_df['content'].astype(str).tolist()[:30]
        combined_pos = " | ".join([r[:150] for r in pos_reviews if len(r) > 20])
        if summarizer and combined_pos:
            report['positive_summary'] = summarize_text(f"Users love: {combined_pos}", summarizer, max_len=150) or ""
        for review in pos_reviews[:3]:
            if len(review) > 30: report['top_praises'].append(review[:200])
    
    category_labels = {'Login/Account Issues': 'login_summary', 'Performance/Bugs': 'bugs_summary',
                       'Feature Requests': 'features_summary', 'Notifications/Messaging': 'messaging_summary'}
    for cluster_id, info in cluster_info.items():
        label = info.get('label', '')
        if label in category_labels:
            texts = info.get('all_texts', [])[:25]
            if texts:
                combined = " | ".join([str(t)[:150] for t in texts if len(str(t)) > 20])
                if summarizer and combined:
                    key = category_labels[label]
                    report[key] = summarize_text(f"Issues about {label}: {combined}", summarizer, max_len=150) or f"{len(texts)} reviews"
    
    if report['negative_summary']:
        report['action_items'].append(f"🔴 Address: {report['negative_summary'][:150]}...")
    for label, key in category_labels.items():
        if report.get(key):
            icons = {'login_summary': '🔐', 'bugs_summary': '🐛', 'features_summary': '✨', 'messaging_summary': '💬'}
            report['action_items'].append(f"{icons.get(key, '📌')} {label}: {report[key][:120]}...")
    
    return report


# ============== DATA PROCESSING ==============

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.,!?\'\"-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def score_to_sentiment(score):
    try:
        score = int(score)
        return 'negative' if score <= 2 else ('neutral' if score == 3 else 'positive')
    except: return 'neutral'

def extract_keywords(texts, top_n=5):
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'it', 'its', 'this', 'that', 'i', 'me', 'my', 'you', 'your', 'we', 'they', 'app', 'not', 'just', 'very', 'much', 'really', 'can', 'get', 'use', 'one'}
    word_counts = Counter()
    for text in texts:
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', str(text).lower()) if w not in stopwords]
        word_counts.update(words)
    return [word for word, _ in word_counts.most_common(top_n)]

def simple_cluster(texts, original_reviews, n_clusters=6):
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
    labels = ['Login/Account Issues', 'Performance/Bugs', 'Updates/Installation', 'Notifications/Messaging', 'Ads/Spam/Privacy', 'Feature Requests']
    for i in range(n_clusters):
        if clusters[i]['cleaned']:
            cluster_info[i] = {'label': labels[i] if i < len(labels) else f'Topic {i}', 'keywords': extract_keywords(clusters[i]['cleaned']),
                              'count': len(clusters[i]['cleaned']), 'all_texts': clusters[i]['cleaned'], 'original_texts': clusters[i]['original']}
    return cluster_info

def process_data(df, sentiment_filter="All", app_filter="All"):
    df = df.copy()
    if 'content' not in df.columns:
        text_cols = [col for col in df.columns if any(x in col.lower() for x in ['text', 'review', 'content'])]
        if text_cols: df['content'] = df[text_cols[0]]
        else: return None
    
    df = df.dropna(subset=['content'])
    df = df[df['content'].astype(str).str.strip() != ''].copy()
    df['cleaned_content'] = df['content'].apply(clean_text)
    df = df[df['cleaned_content'].str.strip() != ''].copy()
    df['sentiment'] = df['score'].apply(score_to_sentiment) if 'score' in df.columns else 'neutral'
    
    if app_filter != "All" and 'app_id' in df.columns:
        df = df[df['app_id'] == app_filter].copy()
    
    total = len(df)
    if total == 0: return None
    
    pos, neg, neu = len(df[df['sentiment'] == 'positive']), len(df[df['sentiment'] == 'negative']), len(df[df['sentiment'] == 'neutral'])
    negative_df = df[df['sentiment'] == 'negative']
    cluster_info = simple_cluster(negative_df['cleaned_content'].tolist(), negative_df['content'].tolist()) if len(negative_df) > 0 else {}
    
    return {
        'summary': {'total_reviews': total, 'positive_count': pos, 'negative_count': neg, 'neutral_count': neu,
                   'positive_percentage': round(pos/total*100, 1), 'negative_percentage': round(neg/total*100, 1),
                   'health_score': round(100 - (neg/total*100), 1)},
        'top_issues': sorted(cluster_info.values(), key=lambda x: x['count'], reverse=True)[:6],
        'cluster_info': cluster_info, 'processed_df': df
    }

def load_sample_data():
    for path in ['data/Training_Data.csv', './data/Training_Data.csv']:
        if os.path.exists(path):
            try: return pd.read_csv(path)
            except: continue
    return None


# ============== UI COMPONENTS ==============

def render_hero():
    st.markdown("""
    <div class="hero-card">
        <h1>📊 AI Product Feedback Analyzer</h1>
        <p>Turn thousands of user reviews into clear, actionable insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(summary):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📝</div>
            <div class="kpi-value">{summary.get('total_reviews', 0):,}</div>
            <div class="kpi-label">Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">😊</div>
            <div class="kpi-value">{summary.get('positive_percentage', 0)}%</div>
            <div class="kpi-label">Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">😟</div>
            <div class="kpi-value">{summary.get('negative_percentage', 0)}%</div>
            <div class="kpi-label">Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💪</div>
            <div class="kpi-value">{summary.get('health_score', 0)}%</div>
            <div class="kpi-label">Health Score</div>
        </div>
        """, unsafe_allow_html=True)

def render_sentiment_chart(summary):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Sentiment Distribution</h3>", unsafe_allow_html=True)
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[summary.get('positive_count', 0), summary.get('neutral_count', 0), summary.get('negative_count', 0)],
        hole=0.45,
        marker_colors=[COLORS['teal'], COLORS['blue'], COLORS['light_cyan']],
        textinfo='label+percent',
        textfont=dict(color=COLORS['navy'], size=13, family='Inter')
    )])
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=COLORS['light_cyan'], size=12)),
                     margin=dict(t=10, b=10, l=10, r=10), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_issues_chart(top_issues):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3>🎯 Top Complaint Categories</h3>", unsafe_allow_html=True)
    
    if not top_issues:
        st.info("No issues found.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    labels = [issue.get('label', 'Unknown')[:20] for issue in top_issues]
    counts = [issue.get('count', 0) for issue in top_issues]
    colors = [COLORS['light_cyan'], COLORS['teal'], COLORS['blue'], COLORS['dark_blue'], COLORS['teal'], COLORS['light_cyan']]
    
    fig = go.Figure(data=[go.Bar(
        x=counts, y=labels, orientation='h', marker=dict(color=colors[:len(labels)], line=dict(width=0)),
        text=counts, textposition='outside', textfont=dict(color=COLORS['light_cyan'], size=12, family='Inter')
    )])
    fig.update_layout(xaxis=dict(tickfont=dict(color=COLORS['teal']), showgrid=False),
                     yaxis=dict(autorange="reversed", tickfont=dict(color=COLORS['light_cyan'], size=11)),
                     margin=dict(t=10, b=30, l=130, r=50), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_ai_summary(report):
    st.markdown("---")
    st.markdown("<h2>🤖 AI-Generated Insights</h2>", unsafe_allow_html=True)
    
    if report.get('overall_summary'):
        st.markdown(f"""<div class="summary-card"><h4>📋 Overall Summary</h4><p>{report['overall_summary']}</p></div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4>😟 User Complaints</h4>", unsafe_allow_html=True)
        if report.get('negative_summary'):
            st.markdown(f"""<div class="category-card"><p>{report['negative_summary']}</p></div>""", unsafe_allow_html=True)
        for c in report.get('top_complaints', [])[:2]:
            st.markdown(f"""<div class="quote-card">"{c}"</div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>😊 User Praises</h4>", unsafe_allow_html=True)
        if report.get('positive_summary'):
            st.markdown(f"""<div class="category-card"><p>{report['positive_summary']}</p></div>""", unsafe_allow_html=True)
        for p in report.get('top_praises', [])[:2]:
            st.markdown(f"""<div class="quote-card">"{p}"</div>""", unsafe_allow_html=True)
    
    st.markdown("<h4 style='margin-top: 1.5rem;'>📂 Category Breakdown</h4>", unsafe_allow_html=True)
    cats = [('🔐 Login', report.get('login_summary')), ('🐛 Bugs', report.get('bugs_summary')),
            ('✨ Features', report.get('features_summary')), ('💬 Messaging', report.get('messaging_summary'))]
    cols = st.columns(2)
    for i, (title, summary) in enumerate(cats):
        if summary:
            with cols[i % 2]:
                st.markdown(f"""<div class="category-card"><h5>{title}</h5><p>{summary}</p></div>""", unsafe_allow_html=True)
    
    if report.get('action_items'):
        st.markdown("<h4 style='margin-top: 1.5rem;'>🎯 Action Items</h4>", unsafe_allow_html=True)
        for item in report['action_items'][:4]:
            st.markdown(f"""<div class="action-item">{item}</div>""", unsafe_allow_html=True)

def render_category_details(top_issues):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3>📂 Reviews by Category</h3>", unsafe_allow_html=True)
    
    icons = {'Login/Account Issues': '🔐', 'Performance/Bugs': '🐛', 'Updates/Installation': '📥',
             'Notifications/Messaging': '💬', 'Ads/Spam/Privacy': '🚫', 'Feature Requests': '✨'}
    
    for issue in top_issues:
        label, count = issue.get('label', 'Unknown'), issue.get('count', 0)
        all_texts = issue.get('original_texts', issue.get('all_texts', []))
        with st.expander(f"{icons.get(label, '📌')} {label} ({count} reviews)"):
            if issue.get('keywords'):
                st.markdown(f"**🏷️ Keywords:** {', '.join(issue['keywords'])}")
            if all_texts:
                st.dataframe(pd.DataFrame({'#': range(1, len(all_texts)+1), 'Review': [str(t)[:400] for t in all_texts]}),
                           use_container_width=True, hide_index=True, height=350)
    st.markdown('</div>', unsafe_allow_html=True)


# ============== MAIN APP ==============

def main():
    render_hero()
    
    if 'df' not in st.session_state: st.session_state.df = None
    if 'ai_report' not in st.session_state: st.session_state.ai_report = None
    if 'analyzed' not in st.session_state: st.session_state.analyzed = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📂 Data Source")
        data_source = st.radio("Choose:", ["📊 Sample Dataset", "📤 Upload CSV"], index=0, label_visibility="collapsed")
        
        if data_source == "📤 Upload CSV":
            uploaded = st.file_uploader("Upload", type=['csv'], label_visibility="collapsed")
            if uploaded:
                st.session_state.df = pd.read_csv(uploaded)
                st.session_state.ai_report = None
                st.session_state.analyzed = False
                st.success(f"✅ {len(st.session_state.df):,} reviews")
        else:
            if st.session_state.df is None:
                st.session_state.df = load_sample_data()
            if st.session_state.df is not None:
                st.success(f"✅ {len(st.session_state.df):,} reviews")
        
        st.markdown("---")
        st.markdown("## 🔍 Filters")
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
        app_filter = "All"
        if st.session_state.df is not None and 'app_id' in st.session_state.df.columns:
            app_filter = st.selectbox("App", ["All"] + list(st.session_state.df['app_id'].dropna().unique()))
        
        st.markdown("---")
        if st.session_state.df is not None:
            if st.button("🚀 Analyze Reviews", type="primary", use_container_width=True):
                st.session_state.analyzed = True
                st.rerun()
        
        st.markdown("---")
        st.caption("v6.0 • Luna Blue")
    
    # Main Content
    df = st.session_state.df
    
    if df is not None and st.session_state.analyzed:
        with st.spinner("🔍 Analyzing user feedback..."):
            results = process_data(df, sentiment_filter, app_filter)
        
        if results:
            render_kpi_cards(results['summary'])
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1: render_sentiment_chart(results['summary'])
            with col2: render_issues_chart(results['top_issues'])
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🤖 Generate AI Summary", type="primary", use_container_width=True):
                    with st.spinner("🧠 Generating insights..."):
                        st.session_state.ai_report = generate_detailed_ai_summary(results['processed_df'], results['cluster_info'], load_summarizer())
            
            if st.session_state.ai_report:
                render_ai_summary(st.session_state.ai_report)
            
            st.markdown("---")
            render_category_details(results['top_issues'])
            
            st.markdown("---")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f"<h3>📋 All Reviews ({sentiment_filter})</h3>", unsafe_allow_html=True)
            display_df = results['processed_df'].copy()
            if sentiment_filter != "All":
                display_df = display_df[display_df['sentiment'] == sentiment_filter.lower()]
            if len(display_df) > 0:
                st.dataframe(pd.DataFrame({'#': range(1, len(display_df)+1), 'Review': display_df['content'].astype(str).tolist(),
                            'Score': display_df['score'].tolist() if 'score' in display_df.columns else ['N/A']*len(display_df),
                            'Sentiment': display_df['sentiment'].str.title().tolist()}), use_container_width=True, hide_index=True, height=450)
                st.success(f"📊 Showing all {len(display_df)} reviews")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif df is not None and not st.session_state.analyzed:
        st.markdown("""<div class="welcome-card"><h2>👈 Click "Analyze Reviews" to Begin</h2><p>Select your data source, apply filters, then start the analysis.</p></div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Select a data source to get started!")


if __name__ == "__main__":
    main()
