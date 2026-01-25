#!/bin/bash
#
# Publish the quiz page to GCS for static hosting.
#
# This script:
# 1. Uploads audio files to gs://dumparoo/cover_songs/audio/
# 2. Generates published HTML with GCS URLs
# 3. Uploads HTML to gs://dumparoo/cover_songs/index.html
#

set -e

# Change to the script's directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "========================================"
echo "Publishing quiz to GCS..."
echo "========================================"
python quizpage/publish_quiz.py
