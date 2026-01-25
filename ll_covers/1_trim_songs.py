#!/Users/mcmenamin/Repos/sundries/ll_covers/venv/bin/python
"""
Script to trim MP3 files into snippets based on the 'snippets' tab in Google Sheets.

Looks up source MP3 files from the 'songs' tab by matching song title,
then trims to the specified time range and saves with an obfuscated filename.
"""

import os
import sys
import subprocess
import hashlib
import gspread
import google.auth
from pathlib import Path


# Google Sheet ID from the URL
SHEET_ID = '10_oOMSlLbyc9VJfA1fGzZeknr4yQxiyW-hY_cfzyZbA'
SONGS_TAB_NAME = 'songs'
SNIPPETS_TAB_NAME = 'snippets'

# Directory for trimmed MP3 files
TRIMMED_DIR = Path('/Users/mcmenamin/Repos/sundries/ll_covers/data/trimmed')


def generate_snippet_filename(title, start, end):
    """
    Generate an obfuscated filename for a snippet.

    The filename consists of a hash of the title (so snippets from the same song
    share a prefix) followed by start/end times in milliseconds.

    Args:
        title: Song title
        start: Start time in seconds (float)
        end: End time in seconds (float)

    Returns:
        Filename like "a3f2b1c9d0e4_72500_120000.mp3"
    """
    title_hash = hashlib.md5(title.lower().strip().encode()).hexdigest()[:12]
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    return f"{title_hash}_{start_ms}_{end_ms}.mp3"


def build_songs_lookup(worksheet):
    """
    Build a lookup dict from the songs tab: {normalized_title: mp3_filepath}

    Args:
        worksheet: The gspread worksheet for the 'songs' tab

    Returns:
        Dict mapping lowercase/stripped song titles to their MP3 file paths
    """
    all_values = worksheet.get_all_values()

    if not all_values:
        return {}

    headers = all_values[0]

    try:
        title_idx = headers.index('Song title')
    except ValueError:
        print("ERROR: 'Song title' column not found in songs tab!")
        return {}

    try:
        filepath_idx = headers.index('MP3 filepath')
    except ValueError:
        print("ERROR: 'MP3 filepath' column not found in songs tab!")
        return {}

    lookup = {}
    for row in all_values[1:]:
        if title_idx < len(row) and filepath_idx < len(row):
            title = row[title_idx].lower().strip()
            filepath = row[filepath_idx]
            if title and filepath:
                lookup[title] = filepath

    return lookup


def trim_audio(input_path, output_path, start_time, end_time):
    """
    Trim an MP3 file to the specified start and end times using ffmpeg.
    Applies postprocessing: stereo conversion, volume normalization, fade in/out.

    Args:
        input_path: Path to the input MP3 file
        output_path: Path where to save the trimmed MP3
        start_time: Start time in seconds (float)
        end_time: End time in seconds (float, None means to end of file)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate duration
        duration = (end_time - start_time) if end_time is not None else None

        # Build audio filter chain:
        # 1. Convert to stereo (aformat)
        # 2. Normalize volume using EBU R128 loudness normalization (loudnorm)
        # 3. Add short fade in/out for smooth playback (afade)
        fade_duration = 0.1  # 100ms fade
        filters = [
            'aformat=channel_layouts=stereo',  # Force stereo output
            'loudnorm=I=-16:TP=-1.5:LRA=11',   # EBU R128 normalization
        ]

        # Add fade in/out if we know the duration
        if duration is not None and duration > (fade_duration * 2):
            fade_out_start = duration - fade_duration
            filters.append(f'afade=t=in:st=0:d={fade_duration}')
            filters.append(f'afade=t=out:st={fade_out_start:.3f}:d={fade_duration}')

        filter_chain = ','.join(filters)

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ss', f'{start_time:.3f}',  # Start time with ms precision
        ]

        # Add duration if end_time is specified
        if duration is not None:
            cmd.extend(['-t', f'{duration:.3f}'])

        # Audio processing options
        cmd.extend([
            '-af', filter_chain,           # Audio filters
            '-ar', '44100',                # Sample rate: 44.1kHz (CD quality)
            '-ac', '2',                    # Channels: stereo
            '-b:a', '192k',                # Bitrate: 192kbps (good quality)
            '-y',                          # Overwrite output file if exists
            str(output_path)
        ])

        print(f"  Trimming from {start_time:.3f}s to {end_time:.3f}s" if end_time else f"  Trimming from {start_time:.3f}s to end")
        print(f"  Applying: stereo, loudnorm, fade, 192kbps")
        print(f"  Saving to {output_path}")

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        if output_path.exists():
            print(f"  SUCCESS: Trimmed audio saved")
            return True
        else:
            print(f"  ERROR: Output file not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: ffmpeg failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def process_sheet():
    """Process the snippets tab, look up source files from songs tab, trim and update paths."""

    # Ensure trimmed directory exists
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)

    # Use default credentials from gcloud auth login
    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )

    # Create a client and open the spreadsheet
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)

    # Load songs tab and build lookup
    print("Loading songs tab...")
    songs_worksheet = spreadsheet.worksheet(SONGS_TAB_NAME)
    songs_lookup = build_songs_lookup(songs_worksheet)
    print(f"Found {len(songs_lookup)} songs with MP3 paths\n")

    # Load snippets tab
    print("Loading snippets tab...")
    snippets_worksheet = spreadsheet.worksheet(SNIPPETS_TAB_NAME)
    all_values = snippets_worksheet.get_all_values()

    if not all_values:
        print("No data found in snippets tab.")
        return

    # First row is the header
    headers = all_values[0]
    print(f"Headers: {headers}")
    print(f"Total rows: {len(all_values) - 1}\n")

    # Find column indices
    try:
        song_title_idx = headers.index('Song title')
    except ValueError:
        print("ERROR: 'Song title' column not found in snippets tab!")
        print(f"Available columns: {headers}")
        return

    try:
        segment_start_idx = headers.index('Segment start')
    except ValueError:
        print("ERROR: 'Segment start' column not found in snippets tab!")
        print(f"Available columns: {headers}")
        return

    try:
        segment_end_idx = headers.index('Segment End')
    except ValueError:
        print("ERROR: 'Segment End' column not found in snippets tab!")
        print(f"Available columns: {headers}")
        return

    try:
        mp3_segment_idx = headers.index('MP3 Segment')
    except ValueError:
        print("ERROR: 'MP3 Segment' column not found in snippets tab!")
        print(f"Available columns: {headers}")
        return

    # Process each row
    updates_made = []
    skipped_missing = []

    for idx, row in enumerate(all_values[1:], start=2):  # Start at 2 for sheet row number
        # Ensure row has enough columns
        while len(row) < len(headers):
            row.append('')

        song_title = row[song_title_idx] if song_title_idx < len(row) else ''
        segment_start = row[segment_start_idx] if segment_start_idx < len(row) else ''
        segment_end = row[segment_end_idx] if segment_end_idx < len(row) else ''
        mp3_segment = row[mp3_segment_idx] if mp3_segment_idx < len(row) else ''

        print(f"\nRow {idx}:")
        print(f"  Song title: {song_title}")
        print(f"  Segment start: {segment_start}")
        print(f"  Segment end: {segment_end}")
        print(f"  Current mp3 segment: {mp3_segment}")

        # Look up source MP3 from songs tab
        normalized_title = song_title.lower().strip()
        source_filepath = songs_lookup.get(normalized_title)

        if not source_filepath:
            print(f"  WARNING: No source MP3 found for '{song_title}' - skipping")
            skipped_missing.append((idx, song_title))
            continue

        print(f"  Source MP3: {source_filepath}")

        # Check if source MP3 file exists on disk
        if not os.path.exists(source_filepath):
            print(f"  WARNING: Source MP3 file not found on disk - skipping")
            skipped_missing.append((idx, song_title))
            continue

        # Parse timestamps as float seconds
        try:
            start_time = float(segment_start) if segment_start else None
        except ValueError:
            print(f"  SKIP: Invalid segment start time '{segment_start}'")
            continue

        try:
            end_time = float(segment_end) if segment_end else None
        except ValueError:
            print(f"  SKIP: Invalid segment end time '{segment_end}'")
            continue

        if start_time is None:
            print(f"  SKIP: No segment start time")
            continue

        if end_time is None:
            print(f"  SKIP: No segment end time")
            continue

        # Check if trimmed segment already exists
        segment_exists = False
        if mp3_segment and os.path.exists(mp3_segment):
            segment_exists = True
            print(f"  ✓ Trimmed segment already exists")
        else:
            print(f"  ✗ Trimmed segment missing or path empty")

        # If segment doesn't exist, create it
        if not segment_exists:
            # Generate obfuscated output filename
            output_filename = generate_snippet_filename(song_title, start_time, end_time)
            output_path = TRIMMED_DIR / output_filename

            # Trim the audio
            if trim_audio(source_filepath, output_path, start_time, end_time):
                # Update the row in the sheet
                new_filepath = str(output_path)
                snippets_worksheet.update_cell(idx, mp3_segment_idx + 1, new_filepath)
                updates_made.append((idx, new_filepath))
                print(f"  Updated sheet with path: {new_filepath}")

    print("\n" + "=" * 80)
    print(f"Processing complete! {len(updates_made)} snippets trimmed and updated.")

    if updates_made:
        print("\nCreated snippets:")
        for row_num, filepath in updates_made:
            print(f"  Row {row_num}: {filepath}")

    if skipped_missing:
        print(f"\nSkipped {len(skipped_missing)} rows due to missing source files:")
        for row_num, title in skipped_missing:
            print(f"  Row {row_num}: '{title}'")


def run_gcloud_auth():
    """Run gcloud auth command to authenticate."""
    print("\n" + "=" * 80)
    print("Authentication needed! Running gcloud auth command...")
    print("=" * 80 + "\n")

    cmd = [
        'gcloud', 'auth', 'application-default', 'login',
        '--scopes=https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/cloud-platform,openid'
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
            process_sheet()
            break  # Success, exit the loop
        except google.auth.exceptions.DefaultCredentialsError:
            if retry_count == 0:
                # First time, try to authenticate
                if run_gcloud_auth():
                    retry_count += 1
                    continue
                else:
                    sys.exit(1)
            else:
                # Already tried to authenticate once
                print("ERROR: Authentication still failing after gcloud auth.")
                print("Please check your gcloud configuration and try again.")
                sys.exit(1)
        except PermissionError as e:
            # This might be a scope issue
            if 'insufficient authentication scopes' in str(e).lower() or retry_count == 0:
                print("\nERROR: Insufficient authentication scopes!")
                if run_gcloud_auth():
                    retry_count += 1
                    continue
                else:
                    sys.exit(1)
            else:
                import traceback
                print(f"ERROR: {e}")
                print("\nFull traceback:")
                traceback.print_exc()
                sys.exit(1)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            sys.exit(1)
