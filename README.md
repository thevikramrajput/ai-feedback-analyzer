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

A production-style AI system that converts raw user reviews into actionable product insights using NLP and pre-trained models.

**🔗 Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/thevikramrajput/ai-feedback-analyzer)

## 🎯 Problem Statement

Product teams cannot manually read thousands of user reviews. They need an automated system to:
- Understand overall user sentiment
- Discover the main complaint categories
- Prioritize issues based on frequency and impact
- Get AI-generated summaries of what users are saying

## 🚀 Solution

An end-to-end AI pipeline that:
1. **Loads** Google Play Store reviews from CSV
2. **Preprocesses** text (cleaning, filtering English reviews)
3. **Analyzes** sentiment based on star ratings
4. **Clusters** negative reviews to discover complaint patterns
5. **Generates** AI summaries using T5 pre-trained model
6. **Visualizes** everything in a product analytics dashboard

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Sentiment Analysis** | Classify reviews as positive, neutral, or negative |
| **Topic Clustering** | Keyword-based clustering for complaint categories |
| **AI Summarization** | T5 model generates summaries of what users are saying |
| **Category-wise Reviews** | View ALL reviews in each complaint category |
| **Interactive Dashboard** | Professional Streamlit UI with charts and filters |
| **Filter Support** | Filter by sentiment, app, and view all reviews |

## 🤖 AI-Powered Summary

Click "Generate Detailed AI Summary" to get:
- **Overall Summary** - AI-generated summary of all reviews
- **What Users Are Complaining About** - Summary of negative feedback
- **What Users Love** - Summary of positive reviews
- **Category-wise Summaries** - Login issues, bugs, feature requests, messaging
- **Actual User Quotes** - Real complaints and praises
- **Action Items** - Recommendations based on user feedback

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core language |
| Streamlit | Dashboard UI |
| FastAPI | Backend API (local) |
| Pandas, NumPy | Data processing |
| scikit-learn | KMeans clustering |
| Transformers | T5 model for AI summarization |
| Plotly | Interactive charts |

## 📁 Project Structure

```
ai-feedback-analyzer/
├── app.py               # Main Streamlit app (HF Spaces)
├── requirements.txt     # Dependencies
├── data/
│   └── Training_Data.csv
├── backend/             # FastAPI (local dev)
│   ├── app.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── sentiment.py
│   ├── clustering.py
│   └── insights.py
└── frontend/
    └── app.py           # Streamlit (local dev)
```

## ⚡ Quick Start

### Hugging Face Spaces (Recommended)
Visit: https://huggingface.co/spaces/thevikramrajput/ai-feedback-analyzer

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### With Backend API (Optional)

```bash
# Terminal 1: Start API
cd backend
uvicorn app:app --reload --port 8000

# Terminal 2: Start Dashboard
cd frontend
streamlit run app.py
```

## 📊 Dashboard Features

- **KPI Cards** - Total reviews, positive/negative %, health score
- **Sentiment Pie Chart** - Visual sentiment distribution
- **Top Issues Bar Chart** - Most frequent complaint categories
- **AI Summary Section** - Detailed insights from reviews
- **Reviews by Category** - View ALL reviews in each category
- **All Reviews Table** - Complete filterable review list

## 📝 CSV Format

Your CSV should have these columns:
- `content` - Review text (required)
- `score` - Star rating 1-5 (required)
- `userLang` - Language code (optional)
- `app_id` - App identifier (optional)

## 🔮 Future Improvements

- [ ] Multilingual support
- [ ] Time-series trend analysis
- [ ] Competitor comparison
- [ ] Export to PDF/Excel
- [ ] Custom LLM integration
