"""
TimeRevind Video Converter
Standalone script to convert videos to H.264 format
Usage: python convert_videos_h264.py <video_directory>
Example: python convert_videos_h264.py ./videos
"""

import sys
import subprocess
from pathlib import Path
import imageio_ffmpeg


def convert_video(src: Path) -> bool:
    """Convert video to H.264 codec and replace original"""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Temporary file for conversion
    temp_file = src.parent / f"{src.stem}_temp.mp4"

    try:
        print(f"  [CONVERTING] {src.name}...")
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(src),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-an", "-movflags", "+faststart", str(temp_file)],
            capture_output=True, timeout=600
        )

        if result.returncode == 0:
            # Delete original
            src.unlink()

            # Rename temp to original name
            temp_file.rename(src)

            file_size_mb = src.stat().st_size / (1024 * 1024)
            print(f"  ✅ {src.name} (H.264 - {file_size_mb:.1f} MB)")
            return True
        else:
            # Clean up temp file if conversion failed
            if temp_file.exists():
                temp_file.unlink()
            error = result.stderr.decode()[-300:]
            print(f"  ❌ {src.name}: {error}")
            return False

    except subprocess.TimeoutExpired:
        if temp_file.exists():
            temp_file.unlink()
        print(f"  ❌ {src.name}: Timeout (video too large)")
        return False
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        print(f"  ❌ {src.name}: {e}")
        return False


def main():
    # Get directory from command line
    if len(sys.argv) < 2:
        print("━" * 60)
        print("TimeRevind Video Converter - Convert videos to H.264")
        print("━" * 60)
        print("\nUsage: python convert_videos_h264.py <video_directory>")
        print("\nExamples:")
        print("  python convert_videos_h264.py ./videos")
        print("  python convert_videos_h264.py D:\\videos")
        print("\nSupported formats:")
        print("  Input: .mp4, .avi, .mkv, .mov, .flv, .wmv")
        print("  Output: .mp4 (H.264 codec)")
        print("\nBehavior:")
        print("  - Converts video to H.264 codec")
        print("  - Replaces original file (same name)")
        print("  - Deletes old format automatically")
        print("  - No filename changes = no DB conflicts")
        print("  - Saves ~70% storage space")
        print("━" * 60)
        sys.exit(1)

    video_dir = Path(sys.argv[1])

    # Validate directory
    if not video_dir.exists():
        print(f"❌ Error: Directory '{video_dir}' does not exist!")
        sys.exit(1)

    if not video_dir.is_dir():
        print(f"❌ Error: '{video_dir}' is not a directory!")
        sys.exit(1)

    # Find all video files
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')
    all_files = list(video_dir.glob('*'))
    videos = [v for v in all_files if v.is_file() and v.suffix.lower() in video_extensions]

    if not videos:
        print(f"⚠️  No videos found in {video_dir}")
        sys.exit(0)

    print("\n" + "━" * 60)
    print(f"📹 TimeRevind Video Converter")
    print("━" * 60)
    print(f"📂 Directory: {video_dir.absolute()}")
    print(f"📊 Found: {len(videos)} video(s)")
    print("━" * 60 + "\n")

    converted = 0
    failed = 0

    for idx, video_file in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] Processing: {video_file.name}")

        # Convert in place (replaces original)
        if convert_video(video_file):
            converted += 1
        else:
            failed += 1

        print()

    # Summary
    print("━" * 60)
    print("📊 SUMMARY")
    print("━" * 60)
    print(f"✅ Converted: {converted}")
    print(f"❌ Failed:    {failed}")
    print(f"📈 Total:     {len(videos)}")
    print("━" * 60)

    if failed == 0:
        print("✨ All done! Videos are ready for streaming.\n")
    else:
        print(f"⚠️  {failed} video(s) failed to convert. Check logs above.\n")


if __name__ == "__main__":
    main()
