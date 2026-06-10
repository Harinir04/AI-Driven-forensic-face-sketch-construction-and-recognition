"""
Render the sample-code chapter into a single-chapter final-report PDF.

Usage:  python _build_pdf.py
Output: FINAL-PROJECT-REPORT.pdf in the same directory.

Pipeline: markdown → HTML (markdown-it-py) → PDF (headless Chrome).
Adapted from the Crowd-src final-report builder so the typography,
page margins and syntax-tinted code blocks match.
"""

import os
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

HERE     = Path(__file__).resolve().parent
OUT_HTML = HERE / "FINAL-PROJECT-REPORT.html"
OUT_PDF  = HERE / "FINAL-PROJECT-REPORT.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# This report contains only the sample-code chapter. Add more chapter
# files here if other sections (introduction, design, testing, etc.)
# are written later — they will be concatenated in this order.
CHAPTER_FILES = [
    "09-sample-code.md",
]

TITLE = ("Forensic Face Generation and Matching System "
         "— Final Project Report")

CSS = """
@page {
    size: A4;
    margin: 22mm 20mm 22mm 25mm;
}
html { font-size: 12pt; }
body {
    font-family: "Times New Roman", Georgia, serif;
    color: #111;
    line-height: 1.55;
    max-width: 100%;
    margin: 0;
    padding: 0;
    text-align: justify;
}
h1 {
    font-size: 22pt;
    text-align: center;
    margin-top: 0.4em;
    margin-bottom: 0.4em;
    page-break-before: always;
    page-break-after: avoid;
    letter-spacing: 0.5pt;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
    font-size: 16pt;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
}
h3 {
    font-size: 13pt;
    margin-top: 1em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}
h4 {
    font-size: 12pt;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
    page-break-after: avoid;
}
p { margin: 0.4em 0 0.6em 0; orphans: 3; widows: 3; }
ul, ol { margin: 0.4em 0 0.6em 1.4em; }
li { margin-bottom: 0.2em; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 10.5pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #444;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}
th { background: #e6e6e6; font-weight: bold; }
code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 10pt;
    background: #f2f2f2;
    padding: 1px 3px;
    border-radius: 2px;
}
pre {
    font-family: Consolas, "Courier New", monospace;
    font-size: 9pt;
    background: #f7f7f7;
    border: 1px solid #ccc;
    padding: 10px 12px;
    line-height: 1.35;
    overflow: visible;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: auto;
}
pre code { background: transparent; padding: 0; border-radius: 0; font-size: 9pt; }
hr {
    border: none;
    border-top: 1px solid #888;
    margin: 1em 0;
}
strong { font-weight: bold; }
blockquote {
    border-left: 3px solid #888;
    margin: 0.6em 0;
    padding: 0.2em 0.8em;
    color: #333;
    font-style: italic;
}
.chapter { page-break-before: always; }
.chapter:first-of-type { page-break-before: avoid; }
"""


def find_chrome():
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def combine_markdown():
    parts = []
    for name in CHAPTER_FILES:
        path = HERE / name
        if not path.exists():
            print(f"  WARN: missing {name}")
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def render_html(markdown_text: str) -> str:
    md = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable("table")
    body = md.render(markdown_text)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>
{CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def html_to_pdf(html_path: Path, pdf_path: Path, chrome_path: str):
    url = html_path.as_uri()
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--virtual-time-budget=10000",
        url,
    ]
    print("  chrome:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print("  STDOUT:", result.stdout)
        print("  STDERR:", result.stderr)
        raise RuntimeError(f"chrome exited with {result.returncode}")


def main():
    print("[1/4] Combining markdown files...")
    md_text = combine_markdown()
    print(f"       total chars: {len(md_text):,}")

    print("[2/4] Rendering HTML...")
    html = render_html(md_text)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"       wrote {OUT_HTML}")

    print("[3/4] Locating Chrome...")
    chrome = find_chrome()
    if not chrome:
        print("  ERROR: Chrome / Edge not found in standard locations.")
        sys.exit(1)
    print(f"       using {chrome}")

    print("[4/4] Printing to PDF...")
    html_to_pdf(OUT_HTML, OUT_PDF, chrome)

    if OUT_PDF.exists():
        size_kb = OUT_PDF.stat().st_size / 1024
        print(f"\nDONE: {OUT_PDF}  ({size_kb:,.1f} KB)")
    else:
        print("ERROR: PDF was not generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
