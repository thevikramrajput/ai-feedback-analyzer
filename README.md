# AI Product Feedback Analyzer

A production-style AI system that converts raw user reviews into actionable product insights for product managers and growth teams.

## 🎯 Problem Statement

Product teams cannot manually read thousands of user reviews. They need an automated system to:
- Understand overall user sentiment
- Discover the main complaint categories
- Prioritize issues based on frequency and impact
- Get actionable recommendations

## 🚀 Solution

An end-to-end AI pipeline that:
1. **Loads** Google Play Store reviews from CSV
2. **Preprocesses** text (cleaning, filtering English reviews)
3. **Analyzes** sentiment based on star ratings
4. **Clusters** reviews to discover topic patterns
5. **Generates** business-friendly insights
6. **Visualizes** everything in a product analytics dashboard

## 📸 Features

| Feature | Description |
|---------|-------------|
| **Sentiment Analysis** | Automatically classify reviews as positive, neutral, or negative |
| **Topic Discovery** | AI-powered clustering to find main complaint categories |
| **Smart Insights** | Actionable recommendations for product improvement |
| **Interactive Dashboard** | Professional Streamlit UI with charts and filters |
| **REST API** | FastAPI backend for integration with other tools |

## 🏗️ Architecture

```
┌──────────────┐
│  CSV Dataset │
└──────┬───────┘
       ↓
┌──────────────┐
│ Preprocessing│
│ (clean, label│
│ sentiment)   │
└──────┬───────┘
       ↓
┌──────────────┐
│ NLP Engine   │
│ - Sentiment  │
│ - Embeddings │
│ - Clustering │
└──────┬───────┘
       ↓
┌──────────────┐
│ Insight Layer│
│ - Top issues │
│ - Metrics    │
│ - Trends     │
└──────┬───────┘
       ↓
┌──────────────┐
│ API (FastAPI)│
└──────┬───────┘
       ↓
┌──────────────┐
│   UI (Web)   │
│ Dashboard    │
└──────────────┘
```

## 📁 Project Structure

```
Product Review analyzer/
│
├── backend/
│   ├── app.py           # FastAPI main application
│   ├── data_loader.py   # CSV loading and validation
│   ├── preprocess.py    # Text cleaning and labeling
│   ├── sentiment.py     # Sentiment analysis logic
│   ├── clustering.py    # Topic clustering with KMeans
│   └── insights.py      # Product metrics and summaries
│
├── frontend/
│   └── app.py           # Streamlit dashboard
│
├── data/
│   └── Training_Data.csv # Google Play Store reviews
│
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Python 3.9+**
- **FastAPI** - Backend API framework
- **Streamlit** - Dashboard UI
- **Pandas** - Data processing
- **scikit-learn** - KMeans clustering
- **sentence-transformers** - Text embeddings (optional)
- **Plotly** - Interactive charts

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend API

```bash
cd backend
uvicorn app:app --reload --port 8000
```

The API will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

### 3. Start the Dashboard

Open a new terminal:

```bash
cd frontend
streamlit run app.py
```

The dashboard will open at: http://localhost:8501

### 4. Run Analysis

1. Click "🚀 Run Analysis" in the sidebar
2. Wait for the pipeline to complete
3. Explore the insights!

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze` | POST | Run full analysis pipeline |
| `/insights` | GET | Get product insights |
| `/clusters` | GET | Get topic clusters |
| `/sentiment` | GET | Get sentiment breakdown |
| `/reviews` | GET | Get filtered reviews |
| `/report` | GET | Get text report |

## 🎨 Dashboard Features

- **KPI Cards** - Total reviews, positive/negative percentages
- **Sentiment Pie Chart** - Visual sentiment distribution
- **Top Issues Bar Chart** - Most frequent complaint categories
- **Recommendations** - AI-generated action items
- **Issue Details** - Expandable cards with keywords and samples
- **Complaints Table** - Searchable list of user complaints

## 📈 Sample Insights

After running the analysis, you'll see insights like:

```
📊 AI PRODUCT FEEDBACK ANALYSIS REPORT
============================================================

📈 OVERVIEW
   Total Reviews Analyzed: 201
   Positive: 138 (68.7%)
   Neutral: 6
   Negative: 57 (28.4%)
   Product Health Score: 71.6%

🎯 TOP COMPLAINT CATEGORIES

   1. login / account / banned
      Complaints: 15
      Impact: 🔴 Critical - Affects core functionality

   2. slow / loading / download
      Complaints: 12
      Impact: 🟡 Medium - Affects user experience

💡 RECOMMENDATIONS
   • 🔐 Account/login issues are prominent. Review authentication flow and ban policies.
   • ⚡ Performance complaints detected. Consider performance optimization sprint.
```

## 🔮 Future Improvements

- [ ] Multilingual support (analyze non-English reviews)
- [ ] LLM-powered summarization of each cluster
- [ ] Time-series trend analysis
- [ ] Competitor comparison
- [ ] Export to PDF/Excel
- [ ] Slack/Email notifications
- [ ] Real-time data ingestion from Play Store API

