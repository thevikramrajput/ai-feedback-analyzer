"""
FastAPI Backend Application
Exposes REST API endpoints for the AI Product Feedback Analyzer.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn

# Import local modules
from data_loader import load_reviews, validate_dataset, get_dataset_info, DEFAULT_DATA_PATH
from preprocess import preprocess_reviews
from sentiment import get_sentiment_summary
from clustering import cluster_reviews, get_top_issues
from insights import generate_insights, format_insights_report

# Initialize FastAPI app
app = FastAPI(
    title="AI Product Feedback Analyzer",
    description="API for analyzing user reviews and generating product insights",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to store processed data
app_state = {
    "raw_data": None,
    "processed_data": None,
    "cluster_info": None,
    "insights": None,
    "is_analyzed": False
}


class AnalysisConfig(BaseModel):
    """Configuration for analysis request."""
    data_path: Optional[str] = DEFAULT_DATA_PATH
    filter_english: bool = True
    n_clusters: int = 6
    focus_negative: bool = True


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    status: str
    message: str
    summary: Optional[Dict] = None


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Product Feedback Analyzer",
        "version": "1.0.0",
        "is_analyzed": app_state["is_analyzed"]
    }


@app.post("/analyze")
async def run_analysis(config: Optional[AnalysisConfig] = None):
    """
    Run the full analysis pipeline on the dataset.
    
    This endpoint:
    1. Loads the CSV dataset
    2. Preprocesses the reviews
    3. Performs topic clustering
    4. Generates insights
    """
    if config is None:
        config = AnalysisConfig()
    
    try:
        # Step 1: Load data
        print(f"📂 Loading data from: {config.data_path}")
        raw_df = load_reviews(config.data_path)
        validate_dataset(raw_df)
        app_state["raw_data"] = raw_df
        
        # Step 2: Preprocess
        print("🔧 Preprocessing reviews...")
        processed_df = preprocess_reviews(raw_df, filter_english=config.filter_english)
        
        # Step 3: Cluster
        print("🎯 Discovering topics...")
        clustered_df, cluster_info = cluster_reviews(
            processed_df, 
            n_clusters=config.n_clusters,
            focus_negative=config.focus_negative
        )
        app_state["processed_data"] = clustered_df
        app_state["cluster_info"] = cluster_info
        
        # Step 4: Generate insights
        print("💡 Generating insights...")
        insights = generate_insights(clustered_df, cluster_info)
        app_state["insights"] = insights
        app_state["is_analyzed"] = True
        
        return {
            "status": "success",
            "message": "Analysis completed successfully",
            "summary": insights.get("summary", {}),
            "top_issues_count": len(insights.get("top_issues", [])),
            "recommendations_count": len(insights.get("recommendations", []))
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/insights")
async def get_insights():
    """
    Get the generated product insights.
    
    Returns comprehensive insights including:
    - Summary statistics
    - Sentiment analysis
    - Top issues
    - Sample complaints
    - Recommendations
    """
    if not app_state["is_analyzed"]:
        raise HTTPException(
            status_code=400, 
            detail="No analysis has been run yet. Please call /analyze first."
        )
    
    return {
        "status": "success",
        "insights": app_state["insights"]
    }


@app.get("/clusters")
async def get_clusters():
    """
    Get the discovered topic clusters.
    
    Returns information about each cluster including:
    - Cluster label
    - Keywords
    - Sample texts
    - Count
    """
    if not app_state["is_analyzed"]:
        raise HTTPException(
            status_code=400,
            detail="No analysis has been run yet. Please call /analyze first."
        )
    
    cluster_info = app_state["cluster_info"]
    
    # Format for API response
    clusters = []
    for cluster_id, info in cluster_info.items():
        clusters.append({
            "id": cluster_id,
            "label": info.get("label", f"Topic {cluster_id}"),
            "keywords": info.get("keywords", []),
            "count": info.get("count", 0),
            "samples": info.get("sample_texts", [])[:3]
        })
    
    # Sort by count
    clusters.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "status": "success",
        "total_clusters": len(clusters),
        "clusters": clusters
    }


@app.get("/sentiment")
async def get_sentiment():
    """
    Get sentiment analysis results.
    
    Returns:
    - Distribution counts
    - Percentages
    - Health score
    """
    if not app_state["is_analyzed"]:
        raise HTTPException(
            status_code=400,
            detail="No analysis has been run yet. Please call /analyze first."
        )
    
    processed_df = app_state["processed_data"]
    summary = get_sentiment_summary(processed_df)
    
    return {
        "status": "success",
        "sentiment": summary
    }


@app.get("/reviews")
async def get_reviews(
    sentiment: Optional[str] = None,
    cluster_id: Optional[int] = None,
    app_id: Optional[str] = None,
    limit: int = 50
):
    """
    Get filtered reviews.
    
    Query parameters:
    - sentiment: Filter by 'positive', 'negative', or 'neutral'
    - cluster_id: Filter by cluster ID
    - app_id: Filter by app ID
    - limit: Maximum number of reviews to return (default 50)
    """
    if not app_state["is_analyzed"]:
        raise HTTPException(
            status_code=400,
            detail="No analysis has been run yet. Please call /analyze first."
        )
    
    df = app_state["processed_data"].copy()
    
    # Apply filters
    if sentiment:
        df = df[df["sentiment"] == sentiment]
    
    if cluster_id is not None:
        df = df[df["cluster_id"] == cluster_id]
    
    if app_id and "app_id" in df.columns:
        df = df[df["app_id"] == app_id]
    
    # Limit results
    df = df.head(limit)
    
    # Format response
    reviews = []
    for _, row in df.iterrows():
        reviews.append({
            "content": row.get("content", ""),
            "cleaned_content": row.get("cleaned_content", ""),
            "score": int(row.get("score", 0)),
            "sentiment": row.get("sentiment", "unknown"),
            "cluster_id": int(row.get("cluster_id", -1)),
            "cluster_label": row.get("cluster_label", "Unknown"),
            "app_id": row.get("app_id", ""),
            "user": row.get("userName", "Anonymous")
        })
    
    return {
        "status": "success",
        "total": len(reviews),
        "reviews": reviews
    }


@app.get("/report")
async def get_text_report():
    """
    Get a formatted text report of the analysis.
    """
    if not app_state["is_analyzed"]:
        raise HTTPException(
            status_code=400,
            detail="No analysis has been run yet. Please call /analyze first."
        )
    
    report = format_insights_report(app_state["insights"])
    
    return {
        "status": "success",
        "report": report
    }


@app.get("/data/info")
async def get_data_info():
    """
    Get information about the loaded dataset.
    """
    if app_state["raw_data"] is None:
        raise HTTPException(
            status_code=400,
            detail="No data loaded. Please call /analyze first."
        )
    
    info = get_dataset_info(app_state["raw_data"])
    return {
        "status": "success",
        "info": info
    }


if __name__ == "__main__":
    print("🚀 Starting AI Product Feedback Analyzer API...")
    print("📖 API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
