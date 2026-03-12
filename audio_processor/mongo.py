# audio_processor/mongo.py
from pymongo import MongoClient
from django.conf import settings

_client = None

def get_db():
    """Return MongoDB database instance (lazy connection)."""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URL)
    return _client[settings.DB_NAME]