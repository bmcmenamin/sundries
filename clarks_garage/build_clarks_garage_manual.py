#!/usr/bin/env python3
"""
Clark's Garage Shop Manual — PDF Archiver
==========================================
Scrapes the complete Clark's Garage 944 shop manual and compiles it into
two PDF files:

  clarks_garage_manual_LINKED.pdf   — Internal hyperlinks (click TOC / in-text
                                        links to jump between sections)
  clarks_garage_manual_PRINT.pdf    — Print-friendly (link destinations shown
                                        as numbered footnotes on each page)

REQUIREMENTS
    pip install requests beautifulsoup4 weasyprint

USAGE
    python3 build_clarks_garage_manual.py

The two PDFs are written to the same directory as this script.
Fetching ~150 pages + images takes a few minutes; a progress bar is shown.
"""

import sys
import os
import re
import time
import base64
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────

INDEX_URL   = "https://clarks-garage.com/shop-manual/repair-procedure-index4.htm"
BASE_DOMAIN = "clarks-garage.com"
DELAY       = 0.4          # seconds between requests (be polite)
TIMEOUT     = 25           # per-request timeout in seconds
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}

# ── Helpers ──────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)

def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(1.5)
            else:
                print(f"\n  ⚠  Could not fetch {url}: {exc}")
                return None

_image_cache = {}

def fetch_image_data_uri(img_url):
    if img_url in _image_cache:
        return _image_cache[img_url]
    r = fetch(img_url)
    if not r or not r.content:
        _image_cache[img_url] = None
        return None
    ct = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    b64 = base64.b64encode(r.content).decode("ascii")
    uri = f"data:{ct};base64,{b64}"
    _image_cache[img_url] = uri
    return uri

def is_clarks_htm(url):
    return (
        BASE_DOMAIN in url
        and ".htm" in url
        and "under_construction" not in url
    )

def page_key(url):
    """Short stable identifier derived from filename."""
    return url.rstrip("/").split("/")[-1].replace(".htm", "")

def get_title(soup, url):
    """Best-effort page title extraction."""
    tag = soup.find("title")
    if tag:
        t = tag.get_text(" ", strip=True)
        t = re.sub(r"\s*[-–|]\s*Clark'?s?\s+Garage.*$", "", t, flags=re.IGNORECASE).strip()
        if t:
            return t
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return page_key(url).replace("-", " ").title()

# ── Index parsing ─────────────────────────────────────────────────────────────

def parse_index(html):
    """
    Returns a list of (link_text, absolute_url) tuples, in document order,
    de-duplicated.
    Also returns the raw index_soup for TOC reconstruction.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen  = set()
    links = []
    for a in soup.find_all("a", href=True):
        href    = a["href"]
        abs_url = urljoin(INDEX_URL, href)
        if is_clarks_htm(abs_url) and abs_url not in seen:
            seen.add(abs_url)
            links.append((a.get_text(" ", strip=True), abs_url))
    return links, soup

# ── Page content processing ───────────────────────────────────────────────────

def process_page(html, url, for_print):
    """
    Clean page HTML:
    • Embed images as base64 data URIs
    • Rewrite internal links as anchor hrefs  (linked mode)
      OR add footnote markers                 (print mode)
    • Remove scripts, navigation chrome, etc.
    Returns cleaned HTML string.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove junk tags
    for tag in soup.find_all(["script", "noscript", "style", "meta",
                               "link", "iframe", "form"]):
        tag.decompose()

    body = soup.find("body") or soup

    # ── Embed images ──────────────────────────────────────────────────────
    for img in body.find_all("img"):
        src = img.get("src", "").strip()
        if not src:
            img.decompose()
            continue
        abs_src = urljoin(url, src)
        uri = fetch_image_data_uri(abs_src)
        if uri:
            img["src"] = uri
            img["style"] = "max-width:100%;height:auto;display:block;margin:6pt auto;"
            for attr in ("width", "height"):
                if img.has_attr(attr):
                    del img[attr]
        else:
            # Replace broken image with a text note
            note = soup.new_tag("span")
            note.string = f"[image: {src}]"
            note["style"] = "color:#999;font-size:8pt;"
            img.replace_with(note)

    # ── Handle links ──────────────────────────────────────────────────────
    footnotes = []

    for a in body.find_all("a", href=True):
        href    = a["href"]
        abs_href = urljoin(url, href)

        if "under_construction" in abs_href:
            a.replace_with(a.get_text(" ", strip=True))
            continue

        if is_clarks_htm(abs_href):
            if for_print:
                n = len(footnotes) + 1
                footnotes.append((n, a.get_text(" ", strip=True), abs_href))
                marker = soup.new_tag("sup")
                marker.string = f"[{n}]"
                a.insert_after(marker)
                a.replace_with(a.get_text(" ", strip=True))
            else:
                a["href"]  = f"#page-{page_key(abs_href)}"
                a["class"] = "ilink"
        elif href.startswith("mailto:") or href.startswith("http"):
            if for_print and not href.startswith("mailto:"):
                n = len(footnotes) + 1
                footnotes.append((n, a.get_text(" ", strip=True), abs_href))
                marker = soup.new_tag("sup")
                marker.string = f"[{n}]"
                a.insert_after(marker)

    content = "".join(str(c) for c in body.children)

    if for_print and footnotes:
        fn_html  = '<div class="fn-block"><hr/><p class="fn-head">Links on this page:</p><ol class="fn-list">'
        for n, txt, href in footnotes:
            safe_href = href.replace("&", "&amp;")
            fn_html += f'<li>[{n}] {txt} — <span class="fn-url">{safe_href}</span></li>'
        fn_html += "</ol></div>"
        content += fn_html

    return content

# ── TOC reconstruction ────────────────────────────────────────────────────────

def build_toc_html(index_soup, page_keys_set, for_print):
    """Reconstruct the TOC from the index page soup."""
    parts = []
    section_headers = {
        "ENGINE", "FUEL AND IGNITION", "ELECTRICAL",
        "BODY", "TRANSMISSION AND CLUTCH", "BRAKES AND SUSPENSION",
        "TROUBLESHOOTING", "MISCELLANEOUS"
    }

    current_section = None
    in_entry = False

    # Walk the index table cells to pick up section headers and links
    for td in index_soup.find_all("td"):
        # Check if this cell is a section header cell
        bold = td.find("b")
        if bold:
            hdr = bold.get_text(" ", strip=True).upper()
            if hdr in section_headers:
                if in_entry:
                    parts.append("</ul>")
                    in_entry = False
                parts.append(f'<h3 class="toc-section">{bold.get_text(" ", strip=True)}</h3>')
                parts.append('<ul class="toc-ul">')
                in_entry = True

        for a in td.find_all("a", href=True):
            abs_href = urljoin(INDEX_URL, a["href"])
            if not is_clarks_htm(abs_href):
                continue
            pkey = page_key(abs_href)
            if pkey not in page_keys_set:
                continue
            txt = a.get_text(" ", strip=True)
            if for_print:
                parts.append(f'<li class="toc-entry">{txt}</li>')
            else:
                parts.append(
                    f'<li class="toc-entry"><a href="#page-{pkey}">{txt}</a></li>'
                )

    if in_entry:
        parts.append("</ul>")

    return "\n".join(parts)

# ── Full document assembly ────────────────────────────────────────────────────

COMMON_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
}
@page {
    size: letter;
    margin: 0.85in 0.9in 0.85in 0.9in;
    @bottom-left  { content: "Clark's Garage Shop Manual"; font-size: 8pt; color: #888; }
    @bottom-right { content: counter(page); font-size: 8pt; color: #888; }
}

/* ── Cover / TOC page ── */
.cover {
    page-break-after: always;
    padding-top: 60pt;
    text-align: center;
}
.cover h1 {
    font-size: 28pt;
    color: #7a0000;
    margin-bottom: 8pt;
}
.cover .subtitle { font-size: 13pt; color: #444; margin-bottom: 4pt; }
.cover .edition  { font-size: 9pt; color: #888; margin: 8pt 0 24pt; }
.cover .credit   { font-size: 8pt; color: #aaa; margin-top: 40pt; }

.toc-container { padding: 0 0 24pt; }
.toc-container h2 { font-size: 15pt; color: #7a0000; margin-bottom: 16pt; border-bottom: 1pt solid #ccc; padding-bottom: 4pt; }
.toc-section  { font-size: 11pt; font-weight: bold; color: #333; margin: 14pt 0 4pt; }
.toc-ul       { list-style: none; padding-left: 12pt; }
.toc-entry    { padding: 1.5pt 0; font-size: 9.5pt; }
.toc-entry a  { color: #003399; text-decoration: none; }
.toc-entry a:hover { text-decoration: underline; }

/* ── Procedure pages ── */
.proc {
    page-break-before: always;
    padding-top: 4pt;
}
h1.proc-title {
    font-size: 14pt;
    color: #7a0000;
    border-bottom: 1.5pt solid #7a0000;
    padding-bottom: 3pt;
    margin-bottom: 10pt;
}
a.ilink { color: #003399; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; }
td, th { padding: 4pt 6pt; vertical-align: top; border: 0; }
img    { max-width: 100%; height: auto; }

/* ── Print footnotes ── */
.fn-block  { margin-top: 18pt; font-size: 8pt; color: #444; }
.fn-head   { font-weight: bold; margin-bottom: 4pt; }
.fn-list   { padding-left: 16pt; }
.fn-list li { margin: 2pt 0; }
.fn-url    { color: #666; word-break: break-all; }
sup        { font-size: 7pt; color: #7a0000; vertical-align: super; }
"""

def build_document(pages_data, index_soup, for_print):
    """Assemble the full HTML string for one edition."""
    edition = "Print Edition" if for_print else "Hyperlinked Edition"
    edition_note = (
        "Link destinations shown as numbered footnotes on each page."
        if for_print else
        "Click any link in the Table of Contents or procedure text to jump to that section."
    )

    page_keys_set = {pk for pk, _, _ in pages_data}
    toc_html = build_toc_html(index_soup, page_keys_set, for_print)

    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Clark's Garage Shop Manual ({edition})</title>
  <style>{COMMON_CSS}</style>
</head>
<body>
"""
    # Cover page
    cover = f"""
<div class="cover">
  <h1>Clark's Garage<br>Shop Manual</h1>
  <p class="subtitle">Porsche 924 / 944 / 944 Turbo / 944 S2 / 968</p>
  <p class="edition">{edition} &mdash; {edition_note}</p>
  <hr style="margin:0 auto;width:60%;border-color:#ccc;">

  <div class="toc-container" style="text-align:left;margin-top:24pt;">
    <h2>Table of Contents</h2>
    {toc_html}
  </div>

  <p class="credit">
    Original content © Clark Fletcher / Clark's Garage 1998–2024<br>
    clarks-garage.com — Archived for preservation.
  </p>
</div>
"""

    # Procedure pages
    proc_parts = []
    for pkey, title, content in pages_data:
        proc_parts.append(
            f'<div class="proc" id="page-{pkey}">'
            f'<h1 class="proc-title">{title}</h1>'
            f'{content}'
            f'</div>'
        )

    return head + cover + "\n".join(proc_parts) + "\n</body></html>"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from weasyprint import HTML as WP_HTML
    except ImportError:
        sys.exit("ERROR: WeasyPrint not installed.  Run:  pip install weasyprint")

    print("=" * 60)
    print("Clark's Garage Shop Manual — PDF Archiver")
    print("=" * 60)

    # 1. Fetch index
    print(f"\nFetching index …")
    r = fetch(INDEX_URL)
    if not r:
        sys.exit("Could not fetch the index page. Check your internet connection.")
    index_html = r.text
    links, index_soup = parse_index(index_html)
    print(f"Found {len(links)} unique procedure pages.\n")

    # 2. Fetch every procedure page
    raw_pages = []   # (pkey, url, title, raw_html)
    for i, (link_text, url) in enumerate(links):
        pkey = page_key(url)
        bar  = f"[{i+1:3d}/{len(links)}]"
        print(f"{bar}  {pkey:<30s}", end=" ", flush=True)
        resp = fetch(url)
        if not resp:
            print("SKIP")
            continue
        soup  = BeautifulSoup(resp.text, "html.parser")
        title = get_title(soup, url)
        raw_pages.append((pkey, url, title, resp.text))
        print(f"✓  {title[:45]}")
        time.sleep(DELAY)

    print(f"\nFetched {len(raw_pages)}/{len(links)} pages.\n")

    # 3. Build both editions
    for for_print in (False, True):
        edition = "PRINT" if for_print else "LINKED"
        print(f"Processing pages for {edition} edition …")

        pages_data = []
        for pkey, url, title, raw_html in raw_pages:
            content = process_page(raw_html, url, for_print)
            pages_data.append((pkey, title, content))

        print(f"  Assembling HTML document …")
        full_html = build_document(pages_data, index_soup, for_print)

        html_path = os.path.join(OUT_DIR, f"clarks_garage_manual_{edition}.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(full_html)
        size_mb = os.path.getsize(html_path) / 1_048_576
        print(f"  HTML written: {html_path}  ({size_mb:.1f} MB)")

        pdf_path = os.path.join(OUT_DIR, f"clarks_garage_manual_{edition}.pdf")
        print(f"  Rendering PDF (this may take a minute) …")
        WP_HTML(filename=html_path).write_pdf(pdf_path)
        size_mb = os.path.getsize(pdf_path) / 1_048_576
        print(f"  PDF written:  {pdf_path}  ({size_mb:.1f} MB)\n")

    print("=" * 60)
    print("Done! Two PDFs created:")
    for edition in ("LINKED", "PRINT"):
        p = os.path.join(OUT_DIR, f"clarks_garage_manual_{edition}.pdf")
        if os.path.exists(p):
            print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
