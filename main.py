import asyncio
import threading
import time
import uvicorn
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def run_twitch_reader():
    """Run the minimal Twitch reader"""
    try:
        print("📺 Starting Twitch reader...")
        from svm import main as reader_main
        reader_main()
    except Exception as e:
        print(f"❌ Reader error: {e}")

def run_api_server():
    """Run the API server"""
    try:
        print("🌐 Starting API server...")
        uvicorn.run(
            "API:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ API error: {e}")

def main():
    print("🚀 Starting Sentiment Analysis System")
    print("=" * 50)
    
    # Check requirements
    required_files = ["vectorizer.sav", "classifier.sav", "svm.py", "API.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            sys.exit(1)
    
    if not os.getenv('TWITCH_CHANNEL'):
        print("❌ TWITCH_CHANNEL not set in environment")
        sys.exit(1)
    
    print("✅ All requirements met")
    
    # Start Twitch reader in separate thread
    reader_thread = threading.Thread(target=run_twitch_reader, daemon=True)
    reader_thread.start()
    
    # Wait a moment for reader to initialize
    time.sleep(3)
    
    # Start API server in separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    print("\n" + "=" * 50)
    print("✅ System started successfully!")
    print("📊 Dashboard API: http://localhost:8000")
    print("📈 API Documentation: http://localhost:8000/docs")
    print("💬 Recent Messages: http://localhost:8000/messages/recent")
    print("📈 Current Sentiment: http://localhost:8000/sentiment/current")
    print("🔴 Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        sys.exit(0)

if __name__ == "__main__":
    main()