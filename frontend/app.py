"""
AI Product Feedback Analyzer - Streamlit Dashboard
A product analytics dashboard for analyzing user reviews.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
import os

# Add backend to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Page configuration
st.set_page_config(
    page_title="AI Product Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - Fixed for visibility
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Fix metric visibility - ensure dark text on light cards */
    [data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #059669 !important;
    }
    
    /* Metric container background */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        color: white !important;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: #e2e8f0 !important;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #f8fafc !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_connection():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_analysis():
    """Trigger the analysis pipeline via API."""
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", timeout=120)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_insights():
    """Fetch insights from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/insights", timeout=30)
        return response.json()
    except Exception as e:
        return None


def get_clusters():
    """Fetch cluster data from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/clusters", timeout=30)
        return response.json()
    except:
        return None


def get_reviews(sentiment=None, limit=50):
    """Fetch reviews from API."""
    try:
        params = {"limit": limit}
        if sentiment:
            params["sentiment"] = sentiment
        response = requests.get(f"{API_BASE_URL}/reviews", params=params, timeout=30)
        return response.json()
    except:
        return None


def load_data_directly():
    """Load and process data directly without API."""
    try:
        from data_loader import load_reviews, DEFAULT_DATA_PATH
        from preprocess import preprocess_reviews
        from clustering import cluster_reviews
        from insights import generate_insights
        
        # Load and process
        raw_df = load_reviews(DEFAULT_DATA_PATH)
        processed_df = preprocess_reviews(raw_df, filter_english=True)
        clustered_df, cluster_info = cluster_reviews(processed_df, n_clusters=6)
        insights = generate_insights(clustered_df, cluster_info)
        
        return {
            "processed_df": clustered_df,
            "cluster_info": cluster_info,
            "insights": insights
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def render_header():
    """Render the main header."""
    st.markdown("""
        <div class="main-header">
            <h1>📊 AI Product Feedback Analyzer</h1>
            <p>Transform user reviews into actionable product insights</p>
        </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(summary):
    """Render KPI metric cards with custom styling."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('total_reviews', 0):,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">📝 Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('positive_percentage', 0)}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">😊 Positive ({summary.get('positive_count', 0):,})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);">
            <div style="font-size: 2.5rem; font-weight: 700;">{summary.get('negative_percentage', 0)}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">😟 Negative ({summary.get('negative_count', 0):,})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        health_score = summary.get('health_score', 0)
        health_color = "#11998e" if health_score >= 70 else "#f5576c" if health_score < 50 else "#ffa726"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {health_color} 0%, #ffffff22 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 2.5rem; font-weight: 700;">{health_score}%</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">💪 Health Score</div>
        </div>
        """, unsafe_allow_html=True)


def render_sentiment_chart(summary):
    """Render sentiment distribution pie chart."""
    st.subheader("📊 Sentiment Distribution")
    
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        summary.get('positive_count', 0),
        summary.get('neutral_count', 0),
        summary.get('negative_count', 0)
    ]
    colors = ['#38ef7d', '#ffa726', '#f5576c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(color='white', size=14)
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(color='white')),
        margin=dict(t=20, b=60, l=20, r=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_top_issues_chart(top_issues):
    """Render top issues bar chart."""
    st.subheader("🎯 Top Complaint Categories")
    
    if not top_issues:
        st.info("No issues discovered yet.")
        return
    
    labels = [issue.get('topic', f"Topic {i}")[:30] for i, issue in enumerate(top_issues)]
    counts = [issue.get('complaint_count', 0) for issue in top_issues]
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts,
            y=labels,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale=[[0, '#ffa726'], [0.5, '#f5576c'], [1, '#d32f2f']],
                showscale=False
            ),
            text=counts,
            textposition='outside',
            textfont=dict(color='white', size=12)
        )
    ])
    
    fig.update_layout(
        xaxis_title="Number of Complaints",
        xaxis=dict(title_font=dict(color='white'), tickfont=dict(color='white')),
        yaxis_title="",
        yaxis=dict(autorange="reversed", tickfont=dict(color='white')),
        margin=dict(t=20, b=40, l=160, r=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_sample_complaints(sample_complaints):
    """Render sample complaints table."""
    st.subheader("📋 Sample Complaints")
    
    if not sample_complaints:
        st.info("No sample complaints available.")
        return
    
    # Create DataFrame for display
    df = pd.DataFrame(sample_complaints)
    
    # Select and rename columns for display
    display_columns = ['content', 'score', 'cluster']
    if 'user' in df.columns:
        display_columns.insert(0, 'user')
    
    available_columns = [col for col in display_columns if col in df.columns]
    df_display = df[available_columns].head(10)
    
    # Truncate long content
    if 'content' in df_display.columns:
        df_display = df_display.copy()
        df_display['content'] = df_display['content'].apply(
            lambda x: str(x)[:150] + '...' if len(str(x)) > 150 else str(x)
        )
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_recommendations(recommendations):
    """Render recommendations section."""
    st.subheader("💡 Recommendations")
    
    if not recommendations:
        st.success("✅ No critical issues found. Keep monitoring!")
        return
    
    for rec in recommendations:
        st.info(rec)


def render_issue_details(top_issues):
    """Render detailed view of top issues."""
    st.subheader("🔍 Issue Details")
    
    if not top_issues:
        return
    
    for i, issue in enumerate(top_issues[:5]):
        with st.expander(f"🔴 {issue.get('topic', f'Issue {i+1}')} ({issue.get('complaint_count', 0)} complaints)"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Keywords:**")
                keywords = issue.get('keywords', [])
                st.write(", ".join(keywords) if keywords else "N/A")
                
                st.markdown("**Impact:**")
                st.write(issue.get('business_impact', 'Unknown'))
            
            with col2:
                st.markdown("**Example Complaints:**")
                samples = issue.get('example_complaints', [])
                for sample in samples[:2]:
                    sample_text = str(sample)
                    st.markdown(f"• _{sample_text[:200]}..._" if len(sample_text) > 200 else f"• _{sample_text}_")


def main():
    """Main dashboard function."""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/analytics.png", width=80)
        st.title("Controls")
        
        # Check API status
        api_connected = check_api_connection()
        
        if api_connected:
            st.success("✅ API Connected")
            use_api = st.checkbox("Use API", value=True)
        else:
            st.warning("⚠️ API Offline")
            st.caption("Start the backend with:")
            st.code("cd backend\nuvicorn app:app --reload")
            use_api = False
        
        st.divider()
        
        # Run Analysis Button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing reviews..."):
                if use_api and api_connected:
                    result = run_analysis()
                    if result.get("status") == "success":
                        st.success("Analysis complete!")
                        st.session_state["analyzed"] = True
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                else:
                    data = load_data_directly()
                    if data:
                        st.session_state["local_data"] = data
                        st.session_state["analyzed"] = True
                        st.success("Analysis complete!")
                        st.rerun()
        
        st.divider()
        
        # Filters
        st.subheader("📌 Filters")
        sentiment_filter = st.selectbox(
            "Sentiment",
            ["All", "Positive", "Neutral", "Negative"]
        )
        
        st.divider()
        st.caption("AI Product Feedback Analyzer v1.0")
    
    # Main content
    if st.session_state.get("analyzed"):
        # Get data
        if use_api and api_connected:
            insights_response = get_insights()
            clusters_response = get_clusters()
            
            if insights_response and insights_response.get("status") == "success":
                insights = insights_response.get("insights", {})
                summary = insights.get("summary", {})
                top_issues = insights.get("top_issues", [])
                sample_complaints = insights.get("sample_complaints", [])
                recommendations = insights.get("recommendations", [])
            else:
                st.error("Failed to fetch insights from API")
                return
        else:
            # Use local data
            local_data = st.session_state.get("local_data", {})
            insights = local_data.get("insights", {})
            summary = insights.get("summary", {})
            top_issues = insights.get("top_issues", [])
            sample_complaints = insights.get("sample_complaints", [])
            recommendations = insights.get("recommendations", [])
        
        # KPI Cards
        render_kpi_cards(summary)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            render_sentiment_chart(summary)
        
        with col2:
            render_top_issues_chart(top_issues)
        
        st.divider()
        
        # Recommendations
        render_recommendations(recommendations)
        
        st.divider()
        
        # Issue Details
        render_issue_details(top_issues)
        
        st.divider()
        
        # Sample Complaints Table
        render_sample_complaints(sample_complaints)
        
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2 style="color: #f8fafc;">👋 Welcome to AI Product Feedback Analyzer</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                This tool helps product teams analyze user reviews and discover actionable insights.
            </p>
            <br>
            <p style="color: #e2e8f0;">Click <b>🚀 Run Analysis</b> in the sidebar to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center;
                        border: 1px solid #475569;">
                <h3 style="color: #f8fafc;">📊 Sentiment Analysis</h3>
                <p style="color: #94a3b8;">Automatically classify reviews as positive, neutral, or negative</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center;
                        border: 1px solid #475569;">
                <h3 style="color: #f8fafc;">🎯 Topic Discovery</h3>
                <p style="color: #94a3b8;">AI-powered clustering to find main complaint categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center;
                        border: 1px solid #475569;">
                <h3 style="color: #f8fafc;">💡 Smart Insights</h3>
                <p style="color: #94a3b8;">Get actionable recommendations for product improvement</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
