#!/Users/mcmenamin/Repos/sundries/ll_covers/venv/bin/python
"""
Script to trim MP3 files based on segment start/end times from Google Sheets.
"""

import os
import sys
import subprocess
import gspread
import google.auth
import re
from pathlib import Path


# Google Sheet ID from the URL
SHEET_ID = '10_oOMSlLbyc9VJfA1fGzZeknr4yQxiyW-hY_cfzyZbA'
TAB_NAME = 'songs'

# Directory for trimmed MP3 files
TRIMMED_DIR = Path('/Users/mcmenamin/Repos/sundries/ll_covers/data/trimmed')


def clean_filename(title):
    """
    Clean a song title to create a safe filename.

    Args:
        title: Song title from the sheet

    Returns:
        Cleaned filename string (lowercase, underscores instead of spaces, etc.)
    """
    if not title:
        return None

    # Convert to lowercase
    cleaned = title.lower()

    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')

    # Remove or replace special characters - keep only alphanumeric, underscores, and hyphens
    cleaned = re.sub(r'[^a-z0-9_-]', '', cleaned)

    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)

    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')

    return cleaned if cleaned else None


def parse_timestamp(timestamp_str):
    """
    Parse a timestamp string to seconds.
    Supports formats: "MM:SS", "HH:MM:SS", or just seconds as a number.

    Args:
        timestamp_str: String timestamp (e.g., "1:30", "0:01:30", "90")

    Returns:
        Float seconds, or None if invalid
    """
    if not timestamp_str or not str(timestamp_str).strip():
        return None

    timestamp_str = str(timestamp_str).strip()

    try:
        # Try parsing as just a number (seconds)
        return float(timestamp_str)
    except ValueError:
        pass

    # Try parsing as MM:SS or HH:MM:SS
    parts = timestamp_str.split(':')
    try:
        if len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        pass

    return None


def trim_audio(input_path, output_path, start_time, end_time):
    """
    Trim an MP3 file to the specified start and end times using ffmpeg.

    Args:
        input_path: Path to the input MP3 file
        output_path: Path where to save the trimmed MP3
        start_time: Start time in seconds
        end_time: End time in seconds (None means to end of file)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ss', str(start_time),  # Start time
        ]

        # Add duration if end_time is specified
        if end_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])  # Duration

        # Output options
        cmd.extend([
            '-c', 'copy',  # Copy codec (fast, no re-encoding)
            '-y',  # Overwrite output file if exists
            str(output_path)
        ])

        print(f"  Trimming from {start_time}s to {end_time}s" if end_time else f"  Trimming from {start_time}s to end")
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
    """Process the Google Sheet, trim MP3s based on segment times, and update paths."""

    # Ensure trimmed directory exists
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)

    # Use default credentials from gcloud auth login
    # Need write access to update the sheet
    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )

    # Create a client and open the spreadsheet
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)

    # Get the specific worksheet/tab
    worksheet = spreadsheet.worksheet(TAB_NAME)

    # Get all values from the worksheet
    all_values = worksheet.get_all_values()

    if not all_values:
        print("No data found in the sheet.")
        return

    # First row is the header
    headers = all_values[0]
    print(f"Headers: {headers}")
    print(f"Total rows: {len(all_values) - 1}\n")

    # Find column indices
    try:
        song_title_idx = headers.index('Song title')
    except ValueError:
        print("ERROR: 'Song title' column not found!")
        print(f"Available columns: {headers}")
        return

    try:
        mp3_filepath_idx = headers.index('MP3 filepath')
    except ValueError:
        print("ERROR: 'MP3 filepath' column not found!")
        print(f"Available columns: {headers}")
        return

    try:
        segment_start_idx = headers.index('Segment start')
    except ValueError:
        print("ERROR: 'Segment start' column not found!")
        print(f"Available columns: {headers}")
        return

    try:
        segment_end_idx = headers.index('Segment End')
    except ValueError:
        print("ERROR: 'Segment End' column not found!")
        print(f"Available columns: {headers}")
        return

    try:
        mp3_segment_idx = headers.index('MP3 Segment')
    except ValueError:
        print("ERROR: 'MP3 Segment' column not found!")
        print(f"Available columns: {headers}")
        return

    # Process each row
    updates_made = []
    for idx, row in enumerate(all_values[1:], start=2):  # Start at 2 for sheet row number
        # Ensure row has enough columns
        while len(row) < len(headers):
            row.append('')

        song_title = row[song_title_idx] if song_title_idx < len(row) else ''
        mp3_filepath = row[mp3_filepath_idx] if mp3_filepath_idx < len(row) else ''
        segment_start = row[segment_start_idx] if segment_start_idx < len(row) else ''
        segment_end = row[segment_end_idx] if segment_end_idx < len(row) else ''
        mp3_segment = row[mp3_segment_idx] if mp3_segment_idx < len(row) else ''

        print(f"\nRow {idx}:")
        print(f"  Song title: {song_title}")
        print(f"  MP3 filepath: {mp3_filepath}")
        print(f"  Segment start: {segment_start}")
        print(f"  Segment end: {segment_end}")
        print(f"  Current MP3 Segment: {mp3_segment}")

        # Check if source MP3 file exists
        if not mp3_filepath or not os.path.exists(mp3_filepath):
            print(f"  SKIP: Source MP3 file not found")
            continue

        # Parse timestamps
        start_time = parse_timestamp(segment_start)
        end_time = parse_timestamp(segment_end)

        if start_time is None:
            print(f"  SKIP: No valid segment start time")
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
            # Generate output filename using cleaned song title and timestamps
            cleaned_title = clean_filename(song_title)
            if cleaned_title:
                base_name = cleaned_title
            else:
                # Fallback to source filename
                source_path = Path(mp3_filepath)
                base_name = source_path.stem

            # Format timestamps for filename (replace : with -)
            start_str = segment_start.replace(':', '-') if segment_start else '0'
            end_str = segment_end.replace(':', '-') if segment_end else 'end'

            output_filename = f"{base_name}__{start_str}_to_{end_str}.mp3"
            output_path = TRIMMED_DIR / output_filename

            # Trim the audio
            if trim_audio(mp3_filepath, output_path, start_time, end_time):
                # Update the row in the sheet
                new_filepath = str(output_path)
                worksheet.update_cell(idx, mp3_segment_idx + 1, new_filepath)
                updates_made.append((idx, new_filepath))
                print(f"  Updated sheet with path: {new_filepath}")

    print("\n" + "=" * 80)
    print(f"Processing complete! {len(updates_made)} files trimmed and updated.")
    if updates_made:
        print("\nUpdated rows:")
        for row_num, filepath in updates_made:
            print(f"  Row {row_num}: {filepath}")


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
