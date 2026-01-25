#!/usr/bin/env python3
"""
Publish the quiz to GCS for static hosting.

This script:
1. Reads quiz_config.json to find all audio files
2. Uploads audio files to gs://dumparoo/cover_songs/audio/
3. Regenerates HTML with GCS base URL
4. Uploads HTML to gs://dumparoo/cover_songs/index.html
"""

import json
import subprocess
import sys
from pathlib import Path

from google.cloud import storage


# GCS Configuration
PROJECT_ID = 'learnings-258303'
BUCKET_NAME = 'dumparoo'
GCS_PREFIX = 'cover_songs'
GCS_AUDIO_PREFIX = f'{GCS_PREFIX}/audio'
GCS_BASE_URL = f'https://storage.googleapis.com/{BUCKET_NAME}/{GCS_AUDIO_PREFIX}/'

# Local paths
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / 'quiz_config.json'
OUTPUT_DIR = SCRIPT_DIR / 'output'
PUBLISHED_HTML_PATH = OUTPUT_DIR / 'quiz_published.html'


def collect_audio_files(config: dict) -> list[Path]:
    """
    Collect all audio file paths from the config.

    Returns list of Path objects for files that exist.
    """
    audio_files = []

    # Instructions sample audio
    sample = config.get('instructions', {}).get('sample_audiofile', '')
    if sample and not sample.startswith(('http://', 'https://', 'gs://')):
        path = (SCRIPT_DIR / sample).resolve()
        if path.exists():
            audio_files.append(path)
        else:
            print(f"  WARNING: Sample audio not found: {sample}")

    # Question audio files
    for question in config.get('questions', []):
        for audiofile in question.get('audiofiles', []):
            if audiofile and not audiofile.startswith(('http://', 'https://', 'gs://')):
                path = Path(audiofile)
                if not path.is_absolute():
                    path = (SCRIPT_DIR / audiofile).resolve()
                if path.exists():
                    audio_files.append(path)
                else:
                    print(f"  WARNING: Audio file not found: {audiofile}")

    return audio_files


def upload_audio_files(bucket, audio_files: list[Path]) -> dict[str, str]:
    """
    Upload audio files to GCS.

    Returns dict mapping local filename to GCS URL.
    Note: Bucket uses uniform bucket-level access, so public access is
    controlled at the bucket level, not per-object.
    """
    uploaded = {}

    for local_path in audio_files:
        filename = local_path.name
        blob_name = f'{GCS_AUDIO_PREFIX}/{filename}'

        print(f"  Uploading {filename}...")
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))

        gcs_url = f'{GCS_BASE_URL}{filename}'
        uploaded[filename] = gcs_url

    return uploaded


def generate_published_html():
    """Generate HTML with GCS base URL."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / 'build_quiz_page.py'),
        str(CONFIG_PATH),
        str(PUBLISHED_HTML_PATH),
        '--base-url', GCS_BASE_URL,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR generating HTML: {result.stderr}")
        return False

    return True


def upload_html(bucket):
    """Upload the published HTML to GCS."""
    blob_name = f'{GCS_PREFIX}/index.html'
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(str(PUBLISHED_HTML_PATH), content_type='text/html')

    return f'https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}'


def main():
    print("=" * 60)
    print("Publishing Quiz to GCS")
    print("=" * 60)

    # Load config
    print("\n1. Loading quiz config...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"   Loaded {len(config.get('questions', []))} questions")

    # Collect audio files
    print("\n2. Collecting audio files...")
    audio_files = collect_audio_files(config)
    print(f"   Found {len(audio_files)} audio files to upload")

    # Initialize GCS client
    print("\n3. Connecting to GCS...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    print(f"   Connected to gs://{BUCKET_NAME}")

    # Upload audio files
    print("\n4. Uploading audio files...")
    if audio_files:
        uploaded = upload_audio_files(bucket, audio_files)
        print(f"   Uploaded {len(uploaded)} files")
    else:
        print("   No audio files to upload")

    # Generate published HTML
    print("\n5. Generating published HTML...")
    if not generate_published_html():
        print("   FAILED to generate HTML")
        sys.exit(1)
    print(f"   Generated: {PUBLISHED_HTML_PATH}")

    # Upload HTML
    print("\n6. Uploading HTML to GCS...")
    public_url = upload_html(bucket)
    print(f"   Uploaded: {public_url}")

    # Done
    print("\n" + "=" * 60)
    print("PUBLISHED SUCCESSFULLY!")
    print(f"\nPublic URL: {public_url}")
    print("=" * 60)


if __name__ == '__main__':
    main()
