#!/bin/bash
#
# Build the quiz page from Google Sheets data.
#
# This script:
# 1. Reads quiz configuration from Google Sheets (songs, snippets, kv_store tabs)
# 2. Generates quiz_config.json
# 3. Builds the static HTML quiz page
#

set -e

# Change to the script's directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo "========================================"
echo "Building quiz config from Google Sheets..."
echo "========================================"
python quizpage/build_quiz_config.py

echo ""
echo "========================================"
echo "Building quiz HTML..."
echo "========================================"
python quizpage/build_quiz_page.py quizpage/quiz_config.json

echo ""
echo "========================================"
echo "Done!"
echo "Output: quizpage/output/quiz.html"
echo "========================================"
