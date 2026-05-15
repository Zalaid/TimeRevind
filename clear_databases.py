"""
Clear all data from Qdrant Cloud and SQLite databases
Run this to start fresh
"""

import os
import sqlite3
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import dotenv

# Load environment variables
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite paths
DB_PATH = "db/timerevind.db"

def clear_sqlite():
    """Clear all tables in SQLite database"""
    logger.info("🗑️  Clearing SQLite database...")

    if not os.path.exists(DB_PATH):
        logger.warning("SQLite database file not found")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Delete all records from tables
        cursor.execute("DELETE FROM activity_events")
        cursor.execute("DELETE FROM events")
        cursor.execute("DELETE FROM person_profiles")
        cursor.execute("DELETE FROM embedding_metadata")

        conn.commit()
        logger.info("✅ SQLite database cleared:")
        logger.info("   - Deleted all activity_events")
        logger.info("   - Deleted all events")
        logger.info("   - Deleted all person_profiles")
        logger.info("   - Deleted all embedding_metadata")

    except Exception as e:
        logger.error(f"Error clearing SQLite: {e}")
        conn.rollback()
    finally:
        conn.close()


def clear_qdrant():
    """Clear/recreate collections in Qdrant Cloud"""
    logger.info("🗑️  Clearing Qdrant Cloud...")

    # Get Qdrant credentials
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY not set in .env file")
        return

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)

        # Get all collections
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]

        # Delete existing collections
        for collection_name in collection_names:
            try:
                client.delete_collection(collection_name)
                logger.info(f"✅ Deleted collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete {collection_name}: {e}")

        # Recreate collections with correct dimensions
        logger.info("📦 Recreating collections...")

        client.create_collection(
            collection_name="body_embeddings",
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
        )
        logger.info("✅ Created body_embeddings collection (2048D)")

        client.create_collection(
            collection_name="face_embeddings",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        logger.info("✅ Created face_embeddings collection (512D from ArcFace)")

    except Exception as e:
        logger.error(f"Error clearing Qdrant: {e}")
        return


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CLEARING ALL DATABASES")
    logger.info("=" * 60)

    clear_sqlite()
    clear_qdrant()

    logger.info("=" * 60)
    logger.info("✅ ALL DATABASES CLEARED - READY FOR FRESH START")
    logger.info("=" * 60)
