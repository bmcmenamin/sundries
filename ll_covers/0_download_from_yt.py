#!/Users/mcmenamin/Repos/sundries/ll_covers/venv/bin/python
"""
Script to download YouTube videos as MP3 files and update Google Sheets.
"""

import os
import sys
import subprocess
import gspread
import google.auth
import time
import re
from pathlib import Path
import yt_dlp

# Google Sheet ID from the URL
SHEET_ID = '10_oOMSlLbyc9VJfA1fGzZeknr4yQxiyW-hY_cfzyZbA'
TAB_NAME = 'songs'

# Directory for downloaded MP3 files
DATA_DIR = Path('/Users/mcmenamin/Repos/sundries/ll_covers/data/raw')


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


def download_youtube_to_mp3(youtube_url, output_path):
    """
    Download a YouTube video as MP3 using yt-dlp.

    Args:
        youtube_url: The YouTube video URL
        output_path: Path where to save the MP3 file

    Returns:
        True if successful, False otherwise
    """
    print(f"  Downloading from YouTube: {youtube_url}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path.with_suffix('')),  # Remove .mp3 extension, yt-dlp will add it
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"  Extracting audio...")
            ydl.download([youtube_url])

        # yt-dlp adds the .mp3 extension, so verify the file exists
        if output_path.exists():
            print(f"  SUCCESS: Downloaded to {output_path}")
            return True
        else:
            print(f"  ERROR: Download completed but file not found at {output_path}")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def process_sheet():
    """Process the Google Sheet, download missing MP3s, and update paths."""

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        youtube_url_idx = headers.index('Youtube Link')
    except ValueError:
        print("ERROR: 'Youtube Link' column not found!")
        print(f"Available columns: {headers}")
        return

    # Process each row
    updates_made = []
    for idx, row in enumerate(all_values[1:], start=2):  # Start at 2 for sheet row number
        # Ensure row has enough columns
        while len(row) < len(headers):
            row.append('')

        song_title = row[song_title_idx] if song_title_idx < len(row) else ''
        youtube_url = row[youtube_url_idx] if youtube_url_idx < len(row) else ''
        mp3_filepath = row[mp3_filepath_idx] if mp3_filepath_idx < len(row) else ''

        print(f"\nRow {idx}:")
        print(f"  Song title: {song_title}")
        print(f"  Youtube Link: {youtube_url}")
        print(f"  Current MP3 path: {mp3_filepath}")

        # Check if MP3 file exists
        mp3_exists = False
        if mp3_filepath and os.path.exists(mp3_filepath):
            mp3_exists = True
            print(f"  ✓ MP3 file exists")
        else:
            print(f"  ✗ MP3 file missing or path empty")

        # If MP3 doesn't exist and we have a YouTube URL, download it
        if not mp3_exists and youtube_url:
            # Generate filename from song title, fallback to video ID or row number
            cleaned_title = clean_filename(song_title)
            if cleaned_title:
                filename = f"{cleaned_title}.mp3"
            else:
                video_id = youtube_url.split('v=')[-1].split('&')[0] if 'v=' in youtube_url else f"video_{idx}"
                filename = f"{video_id}.mp3"

            output_path = DATA_DIR / filename
            print(f"  Using filename: {filename}")

            # Download the file
            if download_youtube_to_mp3(youtube_url, output_path):
                # Update the row in the sheet
                new_filepath = str(output_path)
                worksheet.update_cell(idx, mp3_filepath_idx + 1, new_filepath)
                updates_made.append((idx, new_filepath))
                print(f"  Updated sheet with path: {new_filepath}")

                # Be nice to the API - add a small delay between downloads
                time.sleep(2)
        elif not youtube_url:
            print(f"  SKIP: No Youtube Link provided")

    print("\n" + "=" * 80)
    print(f"Processing complete! {len(updates_made)} files downloaded and updated.")
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
