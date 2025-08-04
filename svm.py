import asyncio
import websockets
import sqlite3
import pickle
import numpy as np
from collections import deque
from datetime import datetime
import os
from dotenv import load_dotenv
import threading

load_dotenv()

class SentimentClassifier:
    def __init__(self, vectorizer_path: str, classifier_path: str):
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        self.classifier = pickle.load(open(classifier_path, 'rb'))
        self.recent_messages = deque(maxlen=1000)
        self.sentiment_cache = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        db = sqlite3.connect('messages.db')
        cursor = db.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS twitch_messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TIMESTAMP,
                user TEXT,
                channel TEXT,
                message TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_sentiments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                sentiment_score REAL,
                processed_at TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES twitch_messages (id)
            )
        """)
        
        db.commit()
        db.close()
        print("✅ Database initialized")
    
    def analyze_sentiment(self, message: str) -> float:
        """Analyze sentiment of a message"""
        message_vector = self.vectorizer.transform([message])
        prediction = self.classifier.predict(message_vector)[0]
        return 1.0 if prediction == 'pos' else -1.0
    
    def store_message(self, user: str, channel: str, message: str) -> int:
        """Store message and return message ID"""
        db = sqlite3.connect('messages.db')
        cursor = db.cursor()
        
        cursor.execute('''
            INSERT INTO twitch_messages (datetime, user, channel, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.utcnow(), user, channel, message))
        
        message_id = cursor.lastrowid
        db.commit()
        db.close()
        
        return message_id
    
    def store_sentiment(self, message_id: int, sentiment_score: float):
        """Store sentiment analysis result"""
        db = sqlite3.connect('messages.db')
        cursor = db.cursor()
        
        cursor.execute("""
            INSERT INTO message_sentiments (message_id, sentiment_score, processed_at)
            VALUES (?, ?, ?)
        """, (message_id, sentiment_score, datetime.utcnow()))
        
        db.commit()
        db.close()
    
    def process_message(self, user: str, channel: str, message: str):
        """Process a complete message"""
        try:
            # Store message
            message_id = self.store_message(user, channel, message)
            
            # Analyze sentiment
            sentiment_score = self.analyze_sentiment(message)
            
            # Store sentiment
            self.store_sentiment(message_id, sentiment_score)
            
            # Update cache
            self.recent_messages.append({
                'user': user,
                'channel': channel,
                'message': message,
                'sentiment': sentiment_score,
                'timestamp': datetime.utcnow().isoformat()
            })
            self.sentiment_cache.append(sentiment_score)
            
            # Print result
            sentiment_label = "POSITIVE" if sentiment_score > 0 else "NEGATIVE"
            print(f"💾 Stored: {user} in #{channel}: {message[:50]}...")
            print(f"💭 Sentiment: {sentiment_label} ({sentiment_score})")
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
        
        def get_average_sentiment(self, n_messages:int = 50) -> dict:
            with self.lock:
                if not self.sentiment_cache:
                    return{
                        'average_sentiment':0,
                        'message_count':0,
                        'timestamp':datetime.utcnow().isoformat()
                    }
                recent_sentiments = self.sentiment_cache[-n_messages:]
                avg_sentiment = np.average(recent_sentiments)

                return {
                    'message':recent_sentiments,
                    'message_count':len(recent_sentiments),
                    'timestamp':datetime.utcnow().isoformat(),
                    'sentiment_distribution':{
                        'positive':sum(1 for s in recent_sentiments if s > 0.1),
                        'neutral':sum(1 for s in recent_sentiments if -0.1<=s<=0.1),
                        'negative':sum(1 for s in recent_sentiments if s < -0.1)
                    }
                }


class AnonymousTwitchReader:
    def __init__(self, channels: list, analyzer: SentimentClassifier):
        self.channels = [ch.lower() for ch in channels]
        self.analyzer = analyzer
        self.message_count = 0
    
    def parse_irc_message(self, raw_message: str):
        """Parse IRC message format"""
        try:
            if "PRIVMSG" in raw_message and not raw_message.startswith("PING"):
                # Format: :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
                parts = raw_message.split(" ", 3)
                if len(parts) >= 4:
                    user_part = parts[0]
                    if user_part.startswith(":"):
                        user = user_part[1:].split("!")[0]
                        channel = parts[2]
                        if channel.startswith("#"):
                            channel = channel[1:]
                        message = parts[3]
                        if message.startswith(":"):
                            message = message[1:]
                        
                        return user, channel, message
        except Exception as e:
            print(f"❌ Error parsing message: {e}")
        
        return None, None, None
    
    async def connect_and_listen(self):
        """Connect to Twitch IRC and listen for messages"""
        uri = "wss://irc-ws.chat.twitch.tv:443"
        
        try:
            print(f"🔗 Connecting to Twitch IRC...")
            async with websockets.connect(uri) as websocket:
                print("✅ Connected to Twitch!")
                
                # Anonymous login (no token required)
                await websocket.send("PASS oauth:justinfan12345")
                await websocket.send("NICK justinfan12345")
                
                # Join channels
                for channel in self.channels:
                    await websocket.send(f"JOIN #{channel}")
                    print(f"📺 Joined #{channel}")
                
                print("🔄 Listening for messages... (Press Ctrl+C to stop)")
                
                # Listen for messages
                async for raw_message in websocket:
                    if "PING" in raw_message:
                        # Respond to ping to keep connection alive
                        await websocket.send("PONG :tmi.twitch.tv")
                        continue
                    
                    user, channel, message = self.parse_irc_message(raw_message)
                    
                    if user and channel and message:
                        self.message_count += 1
                        print(f"\n📨 [{self.message_count}] {user} in #{channel}: {message}")
                        
                        # Process the message
                        self.analyzer.process_message(user, channel, message)
                        
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection closed by Twitch")
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        except Exception as e:
            print(f"❌ Connection error: {e}")

def check_database_content():
    """Check what's in the database"""
    try:
        db = sqlite3.connect('messages.db')
        cursor = db.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM twitch_messages")
        message_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM message_sentiments")
        sentiment_count = cursor.fetchone()[0]
        
        print(f"\n📊 Database Status:")
        print(f"   Messages: {message_count}")
        print(f"   Sentiments: {sentiment_count}")
        
        if message_count > 0:
            cursor.execute("SELECT datetime, user, channel, message FROM twitch_messages ORDER BY datetime DESC LIMIT 3")
            recent = cursor.fetchall()
            print(f"   Recent messages:")
            for msg in recent:
                print(f"     {msg[0]} | {msg[1]} in #{msg[2]}: {msg[3][:30]}...")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Database check error: {e}")

def main():
    """Main function"""
    print("🚀 Starting Minimal Twitch Chat Reader")
    print("=" * 50)
    
    # Check if model files exist
    vectorizer_path = "vectorizer.sav"
    classifier_path = "classifier.sav"
    
    if not os.path.exists(vectorizer_path):
        print(f"❌ Vectorizer not found: {vectorizer_path}")
        return
    
    if not os.path.exists(classifier_path):
        print(f"❌ Classifier not found: {classifier_path}")
        return
    
    print(f"✅ Model files found")
    
    # Get channels from environment
    channels_str = os.getenv('TWITCH_CHANNEL', '')
    if not channels_str:
        print("❌ TWITCH_CHANNEL not set in environment")
        print("Please set TWITCH_CHANNEL=channelname in your .env file")
        return
    
    channels = [ch.strip() for ch in channels_str.split(',') if ch.strip()]
    print(f"✅ Channels: {channels}")
    
    # Initialize analyzer
    print("🤖 Initializing sentiment analyzer...")
    analyzer = SentimentClassifier(vectorizer_path, classifier_path)
    
    # Create reader
    reader = AnonymousTwitchReader(channels, analyzer)
    
    # Check current database content
    check_database_content()
    
    # Start listening
    try:
        asyncio.run(reader.connect_and_listen())
    except KeyboardInterrupt:
        print("\n✅ Stopped by user")
    finally:
        # Check database content after running
        print("\n📊 Final database check:")
        check_database_content()

if __name__ == "__main__":
    main()