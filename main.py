#!/usr/bin/env python3
"""
TimeRevind Main Entry Point
Starts the detection and tracking pipeline
"""

import os
import logging
import argparse
import sys
from pathlib import Path

# Fix ONNX Runtime GPU: expose PyTorch's bundled CUDA 12 DLLs (cublasLt64_12, cudnn64_9)
# so SCRFD, ArcFace, and any ONNX model runs on GPU instead of silently falling back to CPU
import torch
os.environ['PATH'] = os.path.join(os.path.dirname(torch.__file__), 'lib') + os.pathsep + os.environ.get('PATH', '')


class ReadableFormatter(logging.Formatter):
    """Custom formatter for readable log output"""

    LEVEL_SYMBOLS = {
        "DEBUG": "🔧",
        "INFO": "✓",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔴",
    }

    MODULE_ALIASES = {
        "src.core.person_tracker": "Tracker",
        "src.core.video_processor": "Video",
        "src.core.embeddings": "Embeddings",
        "src.database.db_init": "Database",
        "src.core.event_logger": "Events",
        "__main__": "System",
    }

    def format(self, record):
        # Skip httpx verbose logs completely (all http11 debug)
        if record.name.startswith("httpx") or record.name.startswith("http11"):
            return None

        timestamp = self.formatTime(record, "%H:%M:%S")
        symbol = self.LEVEL_SYMBOLS.get(record.levelname, "•")
        module = self.MODULE_ALIASES.get(record.name, record.name.split(".")[-1])
        msg = record.getMessage()

        # Shorten long messages
        if len(msg) > 100:
            msg = msg[:97] + "..."

        # Format: [HH:MM:SS] Symbol Module: Message
        log_format = f"[{timestamp}] {symbol} {module}: {msg}"

        return log_format


# Custom handler to skip None formatted logs
class FilteringHandler(logging.StreamHandler):
    def emit(self, record):
        formatted = self.format(record)
        if formatted is not None:
            super().emit(record)

# Setup logging with readable formatter
handler = FilteringHandler()
handler.setFormatter(ReadableFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)

logger = logging.getLogger(__name__)


def initialize_system():
    """Initialize all system components"""
    logger.info("Initializing TimeRevind...")

    try:
        # Import after logging setup
        from src.database.db_init import initialize_databases
        from src.core.video_processor import VideoProcessor

        # Initialize databases
        db_manager, embedding_store = initialize_databases()
        logger.info("✓ Databases initialized")

        # Warmup Qdrant connection before starting camera
        logger.info("Testing Qdrant connection...")
        embedding_store.test_connection()
        logger.info("✓ Qdrant connection ready")

        # Create video processor
        processor = VideoProcessor(db_manager, embedding_store)
        logger.info("✓ Video processor initialized")

        return processor, db_manager, embedding_store

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TimeRevind - AI-Powered Camera Surveillance System"
    )

    parser.add_argument(
        "--mode",
        choices=["camera", "video"],
        default="camera",
        help="Processing mode: camera (live) or video (file)",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0)",
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Video file path (for video mode)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    try:
        # Initialize system
        logger.info("=" * 50)
        logger.info("TimeRevind Starting Up")
        logger.info("=" * 50)

        processor, db_manager, embedding_store = initialize_system()

        logger.info("\n" + "=" * 50)
        logger.info("System Ready")
        logger.info("=" * 50 + "\n")

        # Process based on mode
        if args.mode == "camera":
            logger.info(f"Starting camera stream (camera {args.camera})...")
            logger.info("Press 'q' to quit\n")
            processor.process_camera_stream(camera_id=args.camera)

        elif args.mode == "video":
            if not args.video:
                logger.error("Video file required for video mode (--video)")
                sys.exit(1)

            video_path = Path(args.video)
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                sys.exit(1)

            logger.info(f"Processing video: {video_path}\n")
            processor.process_video_file(str(video_path))

        # Print final statistics
        logger.info("\n" + "=" * 50)
        stats = processor.get_statistics()
        logger.info(f"Processing Statistics:")
        logger.info(f"  Frames processed: {stats['frames_processed']}")
        logger.info(f"  Total people detected: {stats['tracking']['total_people_detected']}")
        logger.info(f"  Total entries logged: {stats['events']['total_entries']}")
        logger.info("=" * 50)

        logger.info("\nTimeRevind shutdown complete")

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean shutdown
        import gc
        gc.collect()


if __name__ == "__main__":
    main()
