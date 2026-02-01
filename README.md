---
title: AI Feedback Analyzer
emoji: 📊
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.28.2"
app_file: app.py
pinned: false
---

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
ai-feedback-analyzer/
│
├── app.py               # Main Streamlit app (HF Spaces)
├── backend/
│   ├── app.py           # FastAPI main application
│   ├── data_loader.py   # CSV loading and validation
│   ├── preprocess.py    # Text cleaning and labeling
│   ├── sentiment.py     # Sentiment analysis logic
│   ├── clustering.py    # Topic clustering with KMeans
│   └── insights.py      # Product metrics and summaries
│
├── frontend/
│   └── app.py           # Streamlit dashboard (local)
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

### 3. Start the Dashboard

```bash
streamlit run app.py
```

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze` | POST | Run full analysis pipeline |
| `/insights` | GET | Get product insights |
| `/clusters` | GET | Get topic clusters |
| `/sentiment` | GET | Get sentiment breakdown |
| `/reviews` | GET | Get filtered reviews |

## 🎨 Dashboard Features

- **KPI Cards** - Total reviews, positive/negative percentages
- **Sentiment Pie Chart** - Visual sentiment distribution
- **Top Issues Bar Chart** - Most frequent complaint categories
- **Recommendations** - AI-generated action items
- **Complaints Table** - Searchable list of user complaints

##  Future Improvements

- [ ] Multilingual support (analyze non-English reviews)
- [ ] LLM-powered summarization of each cluster
- [ ] Time-series trend analysis
- [ ] Competitor comparison
- [ ] Export to PDF/Excel
