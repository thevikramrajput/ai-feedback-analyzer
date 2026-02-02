"""
AI Product Feedback Analyzer - Streamlit Dashboard
Version: 7.0 - Sky Theme UI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import os
from collections import Counter

# ============== SKY THEME COLOR PALETTE ==============
COLORS = {
    'skybound': '#60B1E0',      # Primary accent - Skybound Blue
    'pearl': '#F2EFEE',         # Light background - Frosted Pearl
    'quartz': '#CCCCCC',        # Borders - Cloudy Quartz
    'twilight': '#C0D6E2',      # Section cards - Soft Twilight
    'honey': '#FFE0BC',         # Highlights - Honey Silk
    'tide': '#5B8CA6',          # Accents - Tranquil Tide
    'text_dark': '#2D4A5E',     # Dark text for light backgrounds
    'text_heading': '#1E3A4C',  # Even darker for headings
}

# Page configuration
st.set_page_config(
    page_title="AI Product Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== SKY THEME CSS ==============
st.markdown(f"""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Background - Frosted Pearl */
    .stApp {{
        background-color: {COLORS['pearl']};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    
    /* Sidebar - Soft Twilight with Dark Text */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['twilight']} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS['text_dark']} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {COLORS['text_heading']} !important;
        font-weight: 600;
    }}
    
    /* Hide Streamlit defaults */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Hero Header - Skybound Blue with White Text */
    .hero-card {{
        background: linear-gradient(135deg, {COLORS['skybound']} 0%, {COLORS['tide']} 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(91, 140, 166, 0.3);
    }}
    .hero-card h1 {{
        color: #FFFFFF;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .hero-card p {{
        color: rgba(255,255,255,0.9);
        font-size: 1.15rem;
        margin: 0.8rem 0 0 0;
    }}
    
    /* KPI Cards - Light backgrounds with Dark Text */
    .kpi-card {{
        padding: 1.8rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid {COLORS['quartz']};
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(91, 140, 166, 0.1);
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(91, 140, 166, 0.2);
    }}
    .kpi-honey {{
        background-color: {COLORS['honey']};
    }}
    .kpi-twilight {{
        background-color: {COLORS['twilight']};
    }}
    .kpi-skybound {{
        background: linear-gradient(135deg, {COLORS['skybound']} 0%, {COLORS['tide']} 100%);
    }}
    .kpi-pearl {{
        background-color: {COLORS['pearl']};
        border: 2px solid {COLORS['quartz']};
    }}
    .kpi-value {{
        font-size: 2.8rem;
        font-weight: 700;
        color: {COLORS['text_heading']};
        line-height: 1;
    }}
    .kpi-value-light {{
        color: #FFFFFF;
    }}
    .kpi-label {{
        font-size: 0.95rem;
        color: {COLORS['text_dark']};
        margin-top: 0.6rem;
        font-weight: 500;
    }}
    .kpi-label-light {{
        color: rgba(255,255,255,0.9);
    }}
    .kpi-icon {{
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }}
    
    /* Section Cards - Twilight with Dark Text */
    .section-card {{
        background-color: {COLORS['twilight']};
        padding: 1.8rem;
        border-radius: 14px;
        border: 1px solid {COLORS['quartz']};
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(91, 140, 166, 0.1);
    }}
    .section-card h3 {{
        color: {COLORS['text_heading']} !important;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }}
    
    /* Summary Card - Pearl with Dark Text */
    .summary-card {{
        background-color: {COLORS['pearl']};
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid {COLORS['quartz']};
        border-left: 4px solid {COLORS['skybound']};
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .summary-card h4 {{
        color: {COLORS['text_heading']};
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }}
    .summary-card p {{
        color: {COLORS['text_dark']};
        margin: 0;
        line-height: 1.6;
    }}
    
    /* Quote Card - Honey with Dark Text */
    .quote-card {{
        background-color: {COLORS['honey']};
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 3px solid {COLORS['tide']};
        margin: 0.6rem 0;
        font-style: italic;
        color: {COLORS['text_dark']};
        font-size: 0.95rem;
    }}
    
    /* Category Card - Pearl with Dark Text */
    .category-card {{
        background-color: {COLORS['pearl']};
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid {COLORS['quartz']};
        border-left: 4px solid {COLORS['tide']};
        margin: 0.8rem 0;
    }}
    .category-card h5 {{
        color: {COLORS['text_heading']};
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }}
    .category-card p {{
        color: {COLORS['text_dark']};
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    
    /* Action Items - Twilight with Dark Text */
    .action-item {{
        background-color: {COLORS['twilight']};
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['skybound']};
        margin: 0.5rem 0;
        color: {COLORS['text_dark']};
        font-weight: 500;
    }}
    
    /* Welcome Card - Honey with Dark Text */
    .welcome-card {{
        background-color: {COLORS['honey']};
        padding: 4rem 3rem;
        border-radius: 16px;
        text-align: center;
        margin: 3rem auto;
        max-width: 600px;
        border: 2px solid {COLORS['tide']};
        box-shadow: 0 8px 32px rgba(91, 140, 166, 0.15);
    }}
    .welcome-card h2 {{
        color: {COLORS['text_heading']};
        font-size: 1.8rem;
        margin: 0;
        font-weight: 600;
    }}
    .welcome-card p {{
        color: {COLORS['text_dark']};
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
    }}
    
    /* Buttons - Skybound Blue with White Text */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['skybound']} 0%, {COLORS['tide']} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['tide']} 0%, #4A7A94 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(96, 177, 224, 0.4) !important;
    }}
    
    /* Select boxes - Light with Dark Text */
    .stSelectbox > div > div {{
        background-color: {COLORS['pearl']} !important;
        border: 1px solid {COLORS['quartz']} !important;
        border-radius: 10px !important;
        color: {COLORS['text_dark']} !important;
    }}
    .stSelectbox label {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Radio buttons */
    .stRadio label {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: {COLORS['pearl']};
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed {COLORS['skybound']};
    }}
    [data-testid="stFileUploader"] * {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Expanders - Light background with Dark Text */
    .streamlit-expanderHeader {{
        background-color: {COLORS['honey']} !important;
        border-radius: 10px !important;
        color: {COLORS['text_dark']} !important;
        font-weight: 500 !important;
    }}
    [data-testid="stExpander"] {{
        background-color: {COLORS['pearl']} !important;
        border: 1px solid {COLORS['quartz']} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stExpander"] * {{
        color: {COLORS['text_dark']} !important;
    }}
    [data-testid="stExpander"] summary {{
        background-color: {COLORS['honey']} !important;
        color: {COLORS['text_heading']} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        background-color: {COLORS['twilight']} !important;
    }}
    .stExpander {{
        border: 1px solid {COLORS['quartz']} !important;
        border-radius: 12px !important;
    }}
    
    /* Dataframes */
    [data-testid="stDataFrame"] {{
        background-color: {COLORS['pearl']};
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid {COLORS['quartz']};
    }}
    
    /* Success/Info boxes */
    .stSuccess {{
        background-color: {COLORS['twilight']} !important;
        color: {COLORS['text_dark']} !important;
        border: 1px solid {COLORS['skybound']} !important;
        border-radius: 10px !important;
    }}
    .stInfo {{
        background-color: {COLORS['honey']} !important;
        color: {COLORS['text_dark']} !important;
        border: 1px solid {COLORS['quartz']} !important;
        border-radius: 10px !important;
    }}
    
    /* Dividers */
    hr {{
        border: none;
        height: 1px;
        background-color: {COLORS['quartz']};
        margin: 2rem 0;
    }}
    
    /* All headings - Dark for readability */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text_heading']} !important;
    }}
    
    /* Main text - Dark for readability */
    .stApp p, .stApp span, .stApp label {{
        color: {COLORS['text_dark']};
    }}
    
    /* Caption */
    .stCaption {{
        color: {COLORS['tide']} !important;
    }}
    
    /* Spinner and Loading States - Light Theme */
    .stSpinner > div {{
        border-top-color: {COLORS['skybound']} !important;
    }}
    .stSpinner {{
        color: {COLORS['text_dark']} !important;
    }}
    
    /* Status/Running Elements - Light background */
    [data-testid="stStatusWidget"],
    .stStatusWidget {{
        background-color: {COLORS['twilight']} !important;
        border: 1px solid {COLORS['quartz']} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stStatusWidget"] * {{
        color: {COLORS['text_dark']} !important;
        background-color: transparent !important;
    }}
    
    /* Cache spinner/status */
    [data-testid="stCachedStFunctionWarning"],
    .stCachedStFunctionWarning {{
        background-color: {COLORS['honey']} !important;
        color: {COLORS['text_dark']} !important;
        border-radius: 10px !important;
    }}
    
    /* Running code status bar - Override dark background */
    .stException, .stWarning {{
        background-color: {COLORS['honey']} !important;
        color: {COLORS['text_dark']} !important;
        border-radius: 10px !important;
    }}
    
    /* Toast messages */
    [data-testid="stToast"] {{
        background-color: {COLORS['twilight']} !important;
        color: {COLORS['text_dark']} !important;
        border: 1px solid {COLORS['skybound']} !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background-color: {COLORS['skybound']} !important;
    }}
    .stProgress {{
        background-color: {COLORS['quartz']} !important;
    }}
    
    /* Code blocks - Light theme style */
    code, pre {{
        background-color: {COLORS['twilight']} !important;
        color: {COLORS['text_dark']} !important;
        border-radius: 6px !important;
    }}
    
    /* Status widget (Running load_summarizer() etc) - Force Light Theme */
    [data-testid="stStatus"],
    [data-testid="stStatusWidget"],
    div[data-testid="stStatus"] {{
        background-color: {COLORS['twilight']} !important;
        border: 1px solid {COLORS['quartz']} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stStatus"] > div,
    [data-testid="stStatus"] details,
    [data-testid="stStatus"] summary {{
        background-color: {COLORS['twilight']} !important;
        color: {COLORS['text_dark']} !important;
    }}
    [data-testid="stStatus"] code,
    [data-testid="stStatus"] pre {{
        background-color: {COLORS['pearl']} !important;
        color: {COLORS['tide']} !important;
        border: 1px solid {COLORS['quartz']} !important;
    }}
    
    /* Override any dark backgrounds in status */
    [data-testid="stStatus"] * {{
        background-color: transparent !important;
    }}
    [data-testid="stStatus"] {{
        background-color: {COLORS['twilight']} !important;
    }}
    [data-testid="stStatus"] code {{
        background-color: {COLORS['pearl']} !important;
        color: {COLORS['tide']} !important;
        padding: 4px 8px !important;
    }}
    
    /* Status widget running state */
    [data-testid="stMarkdownContainer"] code {{
        background-color: {COLORS['twilight']} !important;
        color: {COLORS['tide']} !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }}
    
    /* Element container for status */
    [data-testid="element-container"] {{
        color: {COLORS['text_dark']} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ============== AI SUMMARIZATION ==============

@st.cache_resource
def load_summarizer():
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
    
    # Each KPI uses a different color from the palette
    with col1:
        st.markdown(f"""
        <div class="kpi-card kpi-honey">
            <div class="kpi-icon">📝</div>
            <div class="kpi-value">{summary.get('total_reviews', 0):,}</div>
            <div class="kpi-label">Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card kpi-twilight">
            <div class="kpi-icon">😊</div>
            <div class="kpi-value">{summary.get('positive_percentage', 0)}%</div>
            <div class="kpi-label">Positive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card kpi-skybound">
            <div class="kpi-icon">😟</div>
            <div class="kpi-value kpi-value-light">{summary.get('negative_percentage', 0)}%</div>
            <div class="kpi-label kpi-label-light">Negative</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card kpi-pearl">
            <div class="kpi-icon">💪</div>
            <div class="kpi-value">{summary.get('health_score', 0)}%</div>
            <div class="kpi-label">Health Score</div>
        </div>
        """, unsafe_allow_html=True)

def render_sentiment_chart(summary):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Sentiment Distribution</h3>", unsafe_allow_html=True)
    
    # Uses palette colors with dark text for visibility
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[summary.get('positive_count', 0), summary.get('neutral_count', 0), summary.get('negative_count', 0)],
        hole=0.45,
        marker_colors=[COLORS['tide'], COLORS['quartz'], COLORS['skybound']],
        textinfo='label+percent',
        textfont=dict(color=COLORS['text_dark'], size=13, family='Inter')
    )])
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=COLORS['text_dark'], size=12)),
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
    # All 6 colors used for bars
    colors = [COLORS['skybound'], COLORS['honey'], COLORS['tide'], COLORS['twilight'], COLORS['quartz'], COLORS['skybound']]
    
    fig = go.Figure(data=[go.Bar(
        x=counts, y=labels, orientation='h', marker=dict(color=colors[:len(labels)], line=dict(width=0)),
        text=counts, textposition='outside', textfont=dict(color=COLORS['text_dark'], size=12, family='Inter')
    )])
    fig.update_layout(xaxis=dict(tickfont=dict(color=COLORS['text_dark']), showgrid=False),
                     yaxis=dict(autorange="reversed", tickfont=dict(color=COLORS['text_dark'], size=11)),
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
        st.caption("v7.0 • Sky Theme")
    
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
