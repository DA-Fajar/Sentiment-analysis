from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import Optional
import uvicorn
from datetime import datetime
import threading
import time
import sqlite3
from svm import SentimentClassifier

app = FastAPI(title="Twitch Sentiment Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Svelte dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = sqlite3.connect('messages.db',check_same_thread=False)
    try:
        yield db
    finally:
        db.close()

# Global instances
sentiment_analyzer: Optional[SentimentClassifier] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global sentiment_analyzer
    
    # Import your updated classes
    from svm import SentimentClassifier
    
    # Initialize with your model files
    vectorizer_path = "vectorizer.sav"
    classifier_path = "classifier.sav"
    
    sentiment_analyzer = SentimentClassifier(vectorizer_path, classifier_path)
    
    print("Dashboard API started successfully")

@app.get("/")
async def root():
    return {"message": "Twitch Sentiment Dashboard API", "status": "running"}

@app.get("/messages/recent")
async def get_recent_messages(
    n_messages: int = Query(20, ge=1, le=100, description="Number of recent messages to retrieve"),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get recent messages with sentiment scores"""
    if not sentiment_analyzer:
        raise HTTPException(status_code=500, detail="Error connecting to database")
    
    try:
        cursor = db.cursor()
        cursor.execute(
            'SELECT datetime, user, channel, message FROM twitch_messages ORDER BY datetime desc LIMIT ?',
            (n_messages,)
        )
        return cursor.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")

@app.get("/stats")
async def get_dashboard_stats(
    db: sqlite3.Connection = Depends(get_db)
    
):
    """Get overall dashboard statistics"""
    if not db:
        raise HTTPException(status_code=500, detail="Error Connecting to Database")
    
    try:
        cursor = db.cursor()
        cursor.execute(
            'SELECT count(*) as message_count FROM twitch_messages'
        )
        return cursor.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

# Server-Sent Events for real-time updates
@app.get("/sentiment/stream")
async def sentiment_stream():
    """Stream real-time sentiment updates to dashboard"""
    
    async def event_stream():
        while True:
            try:
                if sentiment_analyzer:
                    # Get current sentiment data
                    sentiment_data = sentiment_analyzer.get_average_sentiment(50)
                    
                    # Format as Server-Sent Event
                    data = json.dumps(sentiment_data)
                    yield f"data: {data}\n\n"
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
                await asyncio.sleep(10)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

# Configuration endpoints
@app.post("/config/update")
async def update_config(
    default_message_count: int = Query(50, ge=1, le=1000),
    update_interval: int = Query(5, ge=1, le=60)
):
    """Update dashboard configuration"""
    return {
        "message": "Configuration updated",
        "default_message_count": default_message_count,
        "update_interval": update_interval
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "dashboard_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

# Requirements for this API:

# pip install fastapi uvicorn python-multipart
