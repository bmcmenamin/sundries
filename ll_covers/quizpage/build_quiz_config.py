#!/usr/bin/env python3
"""
Build quiz_config.json from Google Sheets data.

Reads from three tabs:
- kv_store: Configuration key-value pairs (form IDs, quiz title, instructions)
- songs: Question data filtered by INCLUDE column
- snippets: Audio files filtered by INCLUDE column
"""

import json
import subprocess
import sys
from pathlib import Path

import google.auth
import gspread


# Google Sheet ID
SHEET_ID = '10_oOMSlLbyc9VJfA1fGzZeknr4yQxiyW-hY_cfzyZbA'

# Tab names
KV_STORE_TAB = 'kv_store'
SONGS_TAB = 'songs'
SNIPPETS_TAB = 'snippets'

# Output path
OUTPUT_PATH = Path(__file__).parent / 'quiz_config.json'


def is_truthy(value: str) -> bool:
    """Check if a string value is truthy (TRUE, Yes, 1, etc.)."""
    return value.strip().lower() in ('true', 'yes', '1', 'x', 'included')


def extract_clip_timing(filepath: str) -> tuple[int, int]:
    """
    Extract start and end times from a snippet filename.

    Filenames follow the pattern: {hash}_{start_ms}_{end_ms}.mp3
    Example: 2275d0f0b4b5_10000_15123.mp3 -> (10000, 15123)

    Returns (0, 0) if timing cannot be extracted.
    """
    try:
        filename = Path(filepath).stem  # Remove extension
        parts = filename.split('_')
        if len(parts) >= 3:
            start_ms = int(parts[-2])
            end_ms = int(parts[-1])
            return (start_ms, end_ms)
    except (ValueError, IndexError):
        pass
    return (0, 0)


def sort_clips_by_timing(clips: list[str]) -> list[str]:
    """Sort audio clips by start time, then end time."""
    return sorted(clips, key=extract_clip_timing)


def read_kv_store(spreadsheet) -> dict:
    """Read key-value pairs from kv_store tab."""
    worksheet = spreadsheet.worksheet(KV_STORE_TAB)
    all_values = worksheet.get_all_values()

    if len(all_values) < 2:
        return {}

    # Assume first row is header, rest are key-value pairs
    # Or it could be a simple two-column format
    kv = {}
    for row in all_values:
        if len(row) >= 2 and row[0].strip():
            kv[row[0].strip()] = row[1].strip() if len(row) > 1 else ''

    return kv


def read_songs(spreadsheet) -> list[dict]:
    """Read songs from songs tab, filtered by INCLUDE column."""
    worksheet = spreadsheet.worksheet(SONGS_TAB)
    all_values = worksheet.get_all_values()

    if not all_values:
        return []

    headers = all_values[0]

    # Find column indices
    def get_idx(name):
        try:
            return headers.index(name)
        except ValueError:
            return None

    title_idx = get_idx('Song title')
    include_idx = get_idx('INCLUDE')
    original_year_idx = get_idx('original_year')
    cover_year_idx = get_idx('cover_year')
    original_artist_idx = get_idx('original_artist')
    cover_artist_idx = get_idx('cover_artist')

    if title_idx is None:
        print("ERROR: 'Song title' column not found in songs tab!")
        print(f"Available columns: {headers}")
        return []

    if include_idx is None:
        print("WARNING: 'INCLUDE' column not found - including all songs")

    songs = []
    for row in all_values[1:]:
        # Check INCLUDE flag
        if include_idx is not None:
            include_val = row[include_idx] if include_idx < len(row) else ''
            if not is_truthy(include_val):
                continue

        def get_val(idx):
            return row[idx].strip() if idx is not None and idx < len(row) else ''

        song = {
            'title': get_val(title_idx),
            'original_year': get_val(original_year_idx),
            'cover_year': get_val(cover_year_idx),
            'original_artist': get_val(original_artist_idx),
            'cover_artist': get_val(cover_artist_idx),
        }

        if song['title']:
            songs.append(song)

    return songs


def read_snippets(spreadsheet) -> dict[str, list[str]]:
    """
    Read snippets from snippets tab, filtered by INCLUDE column.

    Returns:
        Dict mapping song title (lowercase) to list of MP3 segment paths
    """
    worksheet = spreadsheet.worksheet(SNIPPETS_TAB)
    all_values = worksheet.get_all_values()

    if not all_values:
        return {}

    headers = all_values[0]

    def get_idx(name):
        try:
            return headers.index(name)
        except ValueError:
            return None

    title_idx = get_idx('Song title')
    include_idx = get_idx('INCLUDE')
    mp3_segment_idx = get_idx('MP3 Segment')

    if title_idx is None:
        print("ERROR: 'Song title' column not found in snippets tab!")
        return {}

    if mp3_segment_idx is None:
        print("ERROR: 'MP3 Segment' column not found in snippets tab!")
        return {}

    snippets = {}
    for row in all_values[1:]:
        # Check INCLUDE flag
        if include_idx is not None:
            include_val = row[include_idx] if include_idx < len(row) else ''
            if not is_truthy(include_val):
                continue

        title = row[title_idx].strip().lower() if title_idx < len(row) else ''
        mp3_path = row[mp3_segment_idx].strip() if mp3_segment_idx < len(row) else ''

        if title and mp3_path:
            if title not in snippets:
                snippets[title] = []
            snippets[title].append(mp3_path)

    return snippets


def build_config():
    """Build quiz_config.json from Google Sheets data."""
    # Authenticate
    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )

    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)

    print("Reading kv_store tab...")
    kv = read_kv_store(spreadsheet)
    print(f"  Found {len(kv)} key-value pairs")

    print("Reading songs tab...")
    songs = read_songs(spreadsheet)
    print(f"  Found {len(songs)} included songs")

    print("Reading snippets tab...")
    snippets = read_snippets(spreadsheet)
    print(f"  Found snippets for {len(snippets)} songs")

    # Build questions array
    questions = []
    missing_audio_count = 0

    for idx, song in enumerate(songs, start=1):
        title_lower = song['title'].lower()
        audiofiles = snippets.get(title_lower, [])

        # Sort clips by start time, then end time for consistent ordering
        audiofiles = sort_clips_by_timing(audiofiles)

        if not audiofiles:
            missing_audio_count += 1
            print(f"  WARNING: No audio files for '{song['title']}'")

        question = {
            'question_number': idx,
            'question_title': song['title'],
            'question_text': f"Original: {song['original_year']}; Cover: {song['cover_year']}",
            'audiofiles': audiofiles,  # Empty list means "media missing"
            'answer_text': f"{song['cover_artist']} (Original: {song['original_artist']})",
        }
        questions.append(question)

    # Build config
    # Key names in kv_store use double underscores for google_form fields
    config = {
        'quiz_title': kv.get('quiz_title', 'Quiz'),
        'google_form_id': kv.get('google_form__id', ''),
        'google_form_fields': {
            'question_title': kv.get('google_form__question_title', ''),
            'feedback_text': kv.get('google_form__feedback_text', ''),
        },
        'audio_base_url': kv.get('audio_base_url', ''),
        'instructions': {
            'instructions_text': kv.get('instructions_text', ''),
            'sample_audiofile': kv.get('sample_audiofile', ''),
            'post_sample_text': kv.get('post_sample_text', ''),
        },
        'questions': questions,
    }

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nGenerated: {OUTPUT_PATH}")
    print(f"  {len(questions)} questions")
    if missing_audio_count:
        print(f"  {missing_audio_count} questions with missing audio")


def run_gcloud_auth():
    """Run gcloud auth command to authenticate."""
    print("\n" + "=" * 80)
    print("Authentication needed! Running gcloud auth command...")
    print("=" * 80 + "\n")

    cmd = [
        'gcloud', 'auth', 'application-default', 'login',
        '--scopes=https://www.googleapis.com/auth/spreadsheets.readonly,https://www.googleapis.com/auth/cloud-platform,openid'
    ]

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("Authentication successful! Retrying script...")
            print("=" * 80 + "\n")
            return True
        else:
            print("\nAuthentication failed. Please run the command manually:")
            print(f"  {' '.join(cmd)}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\nAuthentication failed with error: {e}")
        return False
    except FileNotFoundError:
        print("\nERROR: gcloud command not found. Please install Google Cloud SDK:")
        print("  https://cloud.google.com/sdk/docs/install")
        return False


if __name__ == '__main__':
    max_retries = 2
    retry_count = 0

    while retry_count < max_retries:
        try:
            build_config()
            break
        except google.auth.exceptions.DefaultCredentialsError:
            if retry_count == 0:
                if run_gcloud_auth():
                    retry_count += 1
                    continue
                else:
                    sys.exit(1)
            else:
                print("ERROR: Authentication still failing after gcloud auth.")
                sys.exit(1)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            sys.exit(1)
