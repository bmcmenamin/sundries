#!/usr/bin/env python3
"""
Build script for generating static quiz HTML from JSON configuration.

Usage:
    python build_quiz.py quiz_config.json [output.html]
    python build_quiz.py quiz_config.json --validate  # Validate only, no output
    python build_quiz.py quiz_config.json --base-url https://storage.googleapis.com/bucket/

Examples:
    # Local testing (relative audio paths)
    python build_quiz.py quiz_config.json

    # GCS deployment (override base URL)
    python build_quiz.py quiz_config.json output/quiz.html --base-url https://storage.googleapis.com/my-bucket/audio/
"""

import argparse
import json
import sys
from html import escape
from pathlib import Path

import mistune


def validate_config(config: dict) -> list[str]:
    """
    Validate the quiz configuration.

    Returns list of warning messages (empty if valid).
    Raises ValueError for critical errors.
    """
    warnings = []

    # Check required top-level fields
    if "instructions" not in config:
        raise ValueError("Missing required field: instructions")
    if "questions" not in config:
        raise ValueError("Missing required field: questions")

    # Check instructions structure
    instructions = config.get("instructions", {})
    if "instructions_text" not in instructions:
        warnings.append("instructions.instructions_text is missing")

    # Validate questions
    questions = config.get("questions", [])
    if not questions:
        raise ValueError("questions array is empty")

    # Check for duplicate question numbers
    question_numbers = [q.get("question_number") for q in questions]
    seen = set()
    duplicates = []
    for num in question_numbers:
        if num is not None and num in seen:
            duplicates.append(num)
        seen.add(num)
    if duplicates:
        warnings.append(f"Duplicate question numbers: {duplicates}")

    # Check for missing/non-sequential question numbers
    valid_numbers = [n for n in question_numbers if isinstance(n, int)]
    if valid_numbers:
        expected = set(range(1, max(valid_numbers) + 1))
        actual = set(valid_numbers)
        missing = expected - actual
        if missing:
            warnings.append(f"Missing question numbers: {sorted(missing)}")

    # Check each question
    for i, q in enumerate(questions):
        qnum = q.get("question_number", f"index {i}")
        if "question_number" not in q:
            warnings.append(f"Question at index {i} missing question_number")
        if "audiofiles" not in q or not q["audiofiles"]:
            warnings.append(f"Question {qnum} has no audiofiles")
        if "answer_text" not in q:
            warnings.append(f"Question {qnum} missing answer_text")

    return warnings


def resolve_audio_path(path: str, base_url: str) -> str:
    """
    Convert audio path to usable URL.

    - gs://bucket/path -> https://storage.googleapis.com/bucket/path
    - https://... -> pass through unchanged
    - Relative path + base_url -> use just the filename with base_url
    - Relative path alone -> use as-is (for local file:// access)
    """
    if path.startswith("gs://"):
        bucket_path = path[5:]  # Remove 'gs://'
        return f"https://storage.googleapis.com/{bucket_path}"

    if path.startswith(("http://", "https://")):
        return path

    if base_url:
        # When using base_url, extract just the filename since files are
        # uploaded to GCS with flat structure (no directory hierarchy)
        filename = Path(path).name
        return base_url.rstrip("/") + "/" + filename

    return path


def generate_instructions_html(instructions: dict, base_url: str) -> str:
    """Generate HTML for the instructions section."""
    parts = []

    text = instructions.get("instructions_text", "")
    if text:
        # Render markdown to HTML (handles newlines, bold, italic, lists, etc.)
        text_html = mistune.html(text)
        parts.append(f'<div class="instructions-text">{text_html}</div>')

    sample = instructions.get("sample_audiofile")
    if sample:
        audio_url = resolve_audio_path(sample, base_url)
        parts.append(f'''
        <div class="audio-player">
            <label>Sample:</label>
            <audio controls>
                <source src="{escape(audio_url)}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>''')

    post_text = instructions.get("post_sample_text", "")
    if post_text:
        post_text_html = mistune.html(post_text)
        parts.append(f'<div class="post-sample-text">{post_text_html}</div>')

    return "\n".join(parts)


def generate_question_html(question: dict, base_url: str) -> str:
    """Generate HTML for a single question."""
    qnum = question.get("question_number", "?")
    qtext = question.get("question_text", "")
    qtitle = question.get("question_title", f"Question {qnum}")
    audiofiles = question.get("audiofiles", [])
    answer = question.get("answer_text", "")
    notes = question.get("notes", "")

    # Audio players (or "media missing" message)
    if not audiofiles:
        audio_html = '<p class="media-missing">Media missing</p>'
    else:
        audio_html_parts = []
        for i, audiofile in enumerate(audiofiles, 1):
            audio_url = resolve_audio_path(audiofile, base_url)
            label = f"Potential Clip {i}:" if len(audiofiles) > 1 else "Clip:"
            audio_html_parts.append(f'''
        <div class="audio-player">
            <label>{label}</label>
            <audio controls>
                <source src="{escape(audio_url)}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
        </div>''')
        audio_html = "\n".join(audio_html_parts)

    # Escape the title for use in JavaScript string
    qtitle_js = qtitle.replace("\\", "\\\\").replace("'", "\\'")

    # Optional notes section (only if notes exist)
    notes_html = ""
    if notes:
        notes_html = f'''
        <div class="notes-section">
            <button class="spoiler-toggle notes-toggle" onclick="toggleSpoiler(this)">
                Show Notes from Brenton
            </button>
            <div class="spoiler-content hidden">
                <p class="notes">{escape(notes)}</p>
            </div>
        </div>'''

    return f'''
    <div class="question" data-question-number="{qnum}" data-question-title="{escape(qtitle)}">
        <h3>{qnum}.&nbsp;{escape(qtext)}</h3>

        <div class="audio-players">
            {audio_html}
        </div>

        <div class="answer-section">
            <button class="spoiler-toggle" onclick="toggleSpoiler(this)">
                Show Answer
            </button>
            <div class="spoiler-content hidden">
                <p class="answer">{escape(answer)}</p>
            </div>
        </div>
        {notes_html}
        <div class="feedback-section">
            <label for="feedback-{qnum}">Feedback:</label>
            <input type="text"
                   id="feedback-{qnum}"
                   class="feedback-input"
                   placeholder="Any notes or feedback for this question?">
            <button class="feedback-submit" onclick="submitFeedback({qnum}, '{qtitle_js}')">
                Submit
            </button>
        </div>
    </div>'''


CSS_STYLES = """
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --background-color: #f5f5f5;
    --card-background: #ffffff;
    --text-color: #333333;
    --border-color: #dddddd;
}

* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--background-color);
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 800px;
    margin: 0 auto;
}

header h1 {
    color: var(--primary-color);
    text-align: center;
    margin-bottom: 30px;
}

.instructions, .question {
    background: var(--card-background);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.instructions {
    border-left: 4px solid var(--secondary-color);
}

.audio-player {
    margin: 15px 0;
}

.audio-player audio {
    width: 100%;
    max-width: 400px;
}

.audio-player label {
    display: block;
    font-weight: bold;
    margin-bottom: 5px;
    color: var(--primary-color);
}

.question h3 {
    margin-top: 0;
    color: var(--primary-color);
}

.media-missing {
    color: #999;
    font-style: italic;
    padding: 15px;
    background-color: #f0f0f0;
    border-radius: 5px;
    text-align: center;
}

/* Spoiler styling */
.answer-section {
    margin-top: 15px;
}

.spoiler-toggle {
    background-color: var(--secondary-color);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
}

.spoiler-toggle:hover {
    opacity: 0.9;
}

.spoiler-content {
    margin-top: 10px;
    padding: 15px;
    background-color: #e8f4f8;
    border-radius: 5px;
    border-left: 4px solid var(--secondary-color);
}

.spoiler-content.hidden {
    display: none;
}

.answer {
    font-weight: bold;
    color: var(--primary-color);
    margin: 0;
    text-transform: uppercase;
}

/* Notes section */
.notes-section {
    margin-top: 10px;
}

.notes-toggle {
    background-color: #8e44ad;
}

.notes-section .spoiler-content {
    background-color: #f5eef8;
    border-left-color: #8e44ad;
}

.notes {
    color: var(--text-color);
    margin: 0;
    font-style: italic;
}

/* Feedback section */
.feedback-section {
    margin-top: 20px;
    padding-top: 15px;
    border-top: 1px solid var(--border-color);
}

.feedback-section label {
    display: block;
    font-weight: bold;
    margin-bottom: 5px;
    color: var(--primary-color);
}

.feedback-input {
    width: 100%;
    max-width: 400px;
    padding: 8px 12px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 14px;
    margin-bottom: 10px;
}

.feedback-submit {
    background-color: #27ae60;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
}

.feedback-submit:hover {
    opacity: 0.9;
}

footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--border-color);
    color: #888;
    font-size: 0.9em;
}
"""


def generate_javascript(config: dict) -> str:
    """Generate JavaScript for page interactivity."""
    form_id = config.get("google_form_id", "")
    form_fields = config.get("google_form_fields", {})
    entry_qtitle = form_fields.get("question_title", "")
    entry_feedback = form_fields.get("feedback_text", "")
    entry_respondent = form_fields.get("respondent_id", "")

    return f"""
// Spoiler toggle functionality
function toggleSpoiler(button) {{
    const content = button.nextElementSibling;
    const isHidden = content.classList.contains('hidden');

    if (isHidden) {{
        content.classList.remove('hidden');
        button.textContent = 'Hide Answer';
    }} else {{
        content.classList.add('hidden');
        button.textContent = 'Show Answer';
    }}
}}

// Google Forms configuration
const FORM_CONFIG = {{
    formId: '{form_id}',
    fields: {{
        questionTitle: '{entry_qtitle}',
        feedbackText: '{entry_feedback}',
        respondentId: '{entry_respondent}'
    }}
}};

// Read respondent from URL query parameter (e.g., ?respondent=alice)
const urlParams = new URLSearchParams(window.location.search);
const RESPONDENT = urlParams.get('respondent') || 'anonymous';

function submitFeedback(questionNumber, questionTitle) {{
    const input = document.getElementById(`feedback-${{questionNumber}}`);
    const feedbackText = input.value.trim();

    if (!feedbackText) {{
        alert('Please enter some feedback before submitting.');
        return;
    }}

    if (!FORM_CONFIG.formId) {{
        alert('Google Form not configured. Feedback: ' + feedbackText);
        return;
    }}

    // Construct Google Form auto-submit URL (formResponse instead of viewform)
    const baseUrl = `https://docs.google.com/forms/d/e/${{FORM_CONFIG.formId}}/formResponse`;
    const params = new URLSearchParams();

    if (FORM_CONFIG.fields.questionTitle) {{
        params.set(FORM_CONFIG.fields.questionTitle, questionTitle);
    }}
    if (FORM_CONFIG.fields.feedbackText) {{
        params.set(FORM_CONFIG.fields.feedbackText, feedbackText);
    }}
    if (FORM_CONFIG.fields.respondentId) {{
        params.set(FORM_CONFIG.fields.respondentId, RESPONDENT);
    }}

    // Submit via hidden iframe to avoid leaving the page
    const iframe = document.createElement('iframe');
    iframe.name = 'hidden_iframe_' + questionNumber;
    iframe.style.display = 'none';
    document.body.appendChild(iframe);

    const form = document.createElement('form');
    form.method = 'POST';
    form.action = baseUrl;
    form.target = iframe.name;

    for (const [key, value] of params) {{
        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = key;
        input.value = value;
        form.appendChild(input);
    }}

    document.body.appendChild(form);
    form.submit();

    // Clean up after submission
    setTimeout(() => {{
        document.body.removeChild(form);
        document.body.removeChild(iframe);
    }}, 1000);

    // Clear the input and show confirmation
    input.value = '';
    const btn = input.nextElementSibling;
    const originalText = btn.textContent;
    btn.textContent = 'Submitted!';
    btn.style.backgroundColor = '#27ae60';
    setTimeout(() => {{
        btn.textContent = originalText;
    }}, 2000);
}}
"""


def generate_html(config: dict) -> str:
    """Generate complete HTML page from configuration."""
    base_url = config.get("audio_base_url", "")
    title = config.get("quiz_title", "Quiz")

    # Sort questions by question_number
    questions = sorted(
        config["questions"],
        key=lambda q: q.get("question_number", 0)
    )

    # Generate questions HTML
    questions_html = "\n".join(
        generate_question_html(q, base_url) for q in questions
    )

    instructions_html = generate_instructions_html(config["instructions"], base_url)
    javascript = generate_javascript(config)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{escape(title)}</h1>
        </header>

        <section class="instructions">
            {instructions_html}
        </section>

        <section class="questions">
            {questions_html}
        </section>

        <footer>
            <p>Generated quiz page</p>
        </footer>
    </div>
    <script>
{javascript}
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate static quiz HTML from JSON configuration"
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=None,
        help="Output HTML file path (default: output/quiz.html)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate config only, do not generate output"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override audio_base_url from config"
    )

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)

    # Override base URL if provided
    if args.base_url is not None:
        config["audio_base_url"] = args.base_url

    # Validate
    try:
        warnings = validate_config(config)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.validate:
        if warnings:
            print(f"Configuration has {len(warnings)} warning(s).")
        else:
            print("Configuration is valid.")
        sys.exit(0)

    # Generate HTML
    html = generate_html(config)

    # Determine output path
    output_path = args.output_file
    if output_path is None:
        output_path = args.config_file.parent / "output" / "quiz.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated: {output_path}")

    # Print Google Forms setup hint if not configured
    if not config.get("google_form_id"):
        print("\nNote: Google Form not configured. To enable feedback submission:")
        print("1. Create a Google Form with 'Question Title' and 'Feedback' fields")
        print("2. Get pre-fill link to find entry IDs (e.g., entry.123456789)")
        print("3. Add to your config:")
        print('   "google_form_id": "1FAIpQL...",')
        print('   "google_form_fields": {"question_title": "entry.XXX", "feedback_text": "entry.YYY"}')


if __name__ == "__main__":
    main()
