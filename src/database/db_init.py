"""
Database Initialization Module
Sets up SQLite database and Qdrant Cloud for TimeRevind
"""

import sqlite3
import logging
import os
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import SQLITE_DB_PATH
import dotenv

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, db_path=SQLITE_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        try:
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS person_profiles (
                    person_id           TEXT PRIMARY KEY,
                    name                TEXT DEFAULT NULL,
                    first_seen          TEXT NOT NULL,
                    last_seen           TEXT NOT NULL,
                    total_visits        INTEGER DEFAULT 0,
                    total_time_seconds  INTEGER DEFAULT 0,
                    keyframe_path       TEXT,
                    notes               TEXT,
                    gender              TEXT DEFAULT NULL,
                    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT DEFAULT NULL,
                    camera_id   TEXT DEFAULT '0',
                    video_file  TEXT DEFAULT NULL,
                    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS events (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp               TEXT NOT NULL,
                    person_id               TEXT NOT NULL,
                    event_type              TEXT NOT NULL CHECK(event_type IN ('ENTERED','EXITED')),
                    visit_num               INTEGER NOT NULL,
                    session_id              TEXT NOT NULL,
                    video_file              TEXT DEFAULT NULL,
                    video_timestamp_start   INTEGER DEFAULT NULL,
                    created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id)  REFERENCES person_profiles(person_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS activity_events (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp               TEXT NOT NULL,
                    person_id               TEXT NOT NULL,
                    event_type              TEXT NOT NULL CHECK(event_type IN ('POSE','LOCATION','INTERACTION','DURATION')),
                    action                  TEXT NOT NULL,
                    object_id               TEXT DEFAULT NULL,
                    location                TEXT DEFAULT NULL,
                    duration_seconds        INTEGER DEFAULT NULL,
                    confidence              REAL DEFAULT NULL,
                    visit_num               INTEGER DEFAULT NULL,
                    session_id              TEXT DEFAULT NULL,
                    video_file              TEXT DEFAULT NULL,
                    video_timestamp_start   INTEGER DEFAULT NULL,
                    created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id)  REFERENCES person_profiles(person_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       TEXT NOT NULL,
                    embedding_type  TEXT NOT NULL CHECK(embedding_type IN ('face')),
                    first_seen      TEXT DEFAULT NULL,
                    frame_count     INTEGER DEFAULT 0,
                    quality         TEXT DEFAULT 'high',
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(person_id, embedding_type),
                    FOREIGN KEY (person_id) REFERENCES person_profiles(person_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_person       ON events(person_id);
                CREATE INDEX IF NOT EXISTS idx_events_session      ON events(session_id);
                CREATE INDEX IF NOT EXISTS idx_events_type         ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp    ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_person_time  ON events(person_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_activity_person     ON activity_events(person_id);
                CREATE INDEX IF NOT EXISTS idx_activity_session    ON activity_events(session_id);
                CREATE INDEX IF NOT EXISTS idx_activity_type       ON activity_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_activity_action     ON activity_events(action);
                CREATE INDEX IF NOT EXISTS idx_profiles_name       ON person_profiles(name);
                CREATE INDEX IF NOT EXISTS idx_profiles_last_seen  ON person_profiles(last_seen);
                CREATE INDEX IF NOT EXISTS idx_embedding_person    ON embedding_metadata(person_id);
            """)

            conn.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")

            # Migration: add gender column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE person_profiles ADD COLUMN gender TEXT DEFAULT NULL")
                conn.commit()
                logger.info("Migrated: added gender column to person_profiles")
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to ignore

        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def query(self, sql, params=None):
        """Execute SELECT query"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor.fetchall()
        finally:
            conn.close()

    def execute(self, sql, params=None):
        """Execute INSERT/UPDATE/DELETE query"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database execution error: {e}")
            raise
        finally:
            conn.close()


class EmbeddingStore:
    """Manages Qdrant Cloud vector database for embeddings"""

    def __init__(self):
        # Get Qdrant credentials from environment
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in .env file"
            )

        # Initialize Qdrant Cloud client
        try:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30.0
            )
            logger.info(f"Connected to Qdrant Cloud: {qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant Cloud: {e}")
            raise

        # Ensure collections exist
        self._ensure_collections()

    def _ensure_collections(self):
        """Create collections if they don't exist"""
        try:
            # Get all collections
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            # Create face_embeddings collection (512D vectors from ArcFace)
            if "face_embeddings" not in collection_names:
                self.client.create_collection(
                    collection_name="face_embeddings",
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
                logger.info("Created 'face_embeddings' collection (512D from ArcFace)")
            else:
                # Check if existing collection has correct dimension
                coll = self.client.get_collection("face_embeddings")
                if coll.config.params.vectors.size != 512:
                    logger.warning(f"face_embeddings has {coll.config.params.vectors.size}D vectors, expected 512D")
                    logger.info("Recreating face_embeddings collection with 512D...")
                    self.client.delete_collection("face_embeddings")
                    self.client.create_collection(
                        collection_name="face_embeddings",
                        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                    )
                    logger.info("Recreated 'face_embeddings' collection (512D from ArcFace)")

            # Ensure payload indexes exist for filtered searches (person_id + pose)
            for field in ("person_id", "pose"):
                try:
                    self.client.create_payload_index(
                        collection_name="face_embeddings",
                        field_name=field,
                        field_schema="keyword"
                    )
                    logger.debug(f"Payload index ensured: face_embeddings.{field}")
                except Exception:
                    pass  # Already exists — not an error

        except Exception as e:
            logger.error(f"Error ensuring collections: {e}")
            raise

    def test_connection(self):
        """Test Qdrant connection by performing a simple query"""
        try:
            # Test connection by getting collections
            collections = self.client.get_collections()
            logger.info(f"✓ Qdrant connection test successful ({len(collections.collections)} collections)")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {e}")
            raise

    def add_face_embedding(self, person_id, embedding, metadata=None):
        """Store face embedding for a person in Qdrant"""
        if metadata is None:
            metadata = {}

        try:
            # Convert embedding to list if needed
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            metadata.update({
                "person_id": person_id,
                "embedding_type": "face",
                "timestamp": datetime.now().isoformat()
            })

            # Use deterministic hash for point ID with session_id for accumulation across runs
            import hashlib
            embedding_index = metadata.get("embedding_index", 1)
            session_id = metadata.get("session_id", 0)
            point_id = int(hashlib.md5(f"{person_id}_face_{embedding_index}_{session_id}".encode()).hexdigest(), 16) % (2**31)

            point = PointStruct(
                id=point_id,
                vector=embedding_list,
                payload=metadata
            )

            self.client.upsert(
                collection_name="face_embeddings",
                points=[point]
            )
            logger.debug(f"Added face embedding for {person_id} (point_id: {point_id})")
        except Exception as e:
            logger.error(f"Error adding face embedding: {e}")
            raise

    def search_face_embedding(self, embedding, top_k=1):
        """Search for matching face embedding in Qdrant"""
        try:
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            results = self.client.query_points(
                collection_name="face_embeddings",
                query=embedding_list,
                limit=top_k,
                with_payload=True
            )

            # Convert Qdrant format to ChromaDB-like format for compatibility
            # Clamp distance to [0, inf) to avoid negative values from floating point precision
            return {
                "ids": [[r.id] for r in results.points],
                "distances": [[max(0.0, 1 - r.score)] for r in results.points] if results.points else [],
                "metadatas": [[r.payload] for r in results.points] if results.points else []
            }
        except Exception as e:
            logger.warning(f"Error searching face embeddings: {e}")
            return {"ids": [], "distances": [], "metadatas": []}

    def search_batch(self, collection_name, embeddings, top_k=1, pose_filter=None):
        """Batch search — ONE API call for all embeddings.

        pose_filter: optional str or list of pose labels.
                     Single str  → MatchValue (exact match)
                     List        → MatchAny   (any of the listed poses)
                     e.g. ["FRONT","UP","DOWN"] matches the frontal group.
        """
        try:
            from qdrant_client.models import QueryRequest, Filter, FieldCondition, MatchValue, MatchAny

            query_filter = None
            if pose_filter:
                if isinstance(pose_filter, list):
                    query_filter = Filter(
                        must=[FieldCondition(key="pose", match=MatchAny(any=pose_filter))]
                    )
                else:
                    query_filter = Filter(
                        must=[FieldCondition(key="pose", match=MatchValue(value=pose_filter))]
                    )

            requests = [
                QueryRequest(
                    query=emb.tolist() if hasattr(emb, 'tolist') else emb,
                    limit=top_k,
                    with_payload=True,
                    filter=query_filter
                )
                for emb in embeddings
            ]

            results = self.client.query_batch_points(
                collection_name=collection_name,
                requests=requests
            )

            results_list = []
            for batch_result in results:
                points = batch_result.points
                result_dict = {
                    "ids": [[r.id] for r in points],
                    "distances": [[max(0.0, 1 - r.score)] for r in points] if points else [],
                    "metadatas": [[r.payload] for r in points] if points else []
                }
                results_list.append(result_dict)

            return results_list

        except Exception as e:
            logger.warning(f"Batch search failed: {e}")
            return [{"ids": [], "distances": [], "metadatas": []} for _ in embeddings]

    def get_person_embeddings(self, person_id):
        """Get all face embeddings for a person from Qdrant"""
        try:
            face_results = self.client.scroll(
                collection_name="face_embeddings",
                limit=100,
                query_filter={
                    "must": [
                        {
                            "key": "person_id",
                            "match": {"value": person_id}
                        }
                    ]
                }
            )

            return {"face": face_results}
        except Exception as e:
            logger.warning(f"Error getting person embeddings: {e}")
            return {"face": []}

    def get_embeddings_for_person(self, person_id):
        """
        Load all stored face embeddings for a person from Qdrant into memory.
        Falls back to empty list if not found.

        Args:
            person_id: Person ID to fetch embeddings for

        Returns:
            List of numpy arrays (face embeddings) or empty list if not found
        """
        import numpy as np

        try:
            embeddings = []
            offset = 0
            page_size = 100

            while True:
                points, next_offset = self.client.scroll(
                    collection_name="face_embeddings",
                    offset=offset,
                    limit=page_size,
                    with_vectors=True,
                    with_payload=True
                )

                if not points:
                    break

                for point in points:
                    if (point.payload and
                        point.payload.get("person_id") == person_id and
                        point.vector is not None):
                        embeddings.append(np.array(point.vector, dtype=np.float32))

                if len(embeddings) >= 30:
                    embeddings = embeddings[:30]
                    break

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(f"Loaded {len(embeddings)} face embeddings for {person_id}")
            return embeddings

        except Exception as e:
            logger.warning(f"Failed to load face embeddings for {person_id}: {e}")
            return []

    def persist(self):
        """Qdrant Cloud persists automatically"""
        logger.debug("Qdrant Cloud data persisted automatically")


# Global instances (will be initialized in main)
db_manager = None
embedding_store = None


def initialize_databases():
    """Initialize all database components"""
    global db_manager, embedding_store

    db_manager = DatabaseManager()
    embedding_store = EmbeddingStore()

    logger.info("All databases initialized successfully")

    return db_manager, embedding_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_databases()
    print("Database initialization complete!")
