"""Checks the manual PDF: no blank pages, every page has text or an image, no dash bullets.

Usage: python tools/manual/check_manual.py MANUAL.pdf
Exit code 1 when a problem is found.
"""

import re
import sys

from pypdf import PdfReader

MIN_CHARS = 350  # a page with less text must carry an image
problems = []
r = PdfReader(sys.argv[1])
print("pages:", len(r.pages))
for i, page in enumerate(r.pages, 1):
    text = (page.extract_text() or "").strip()
    body = re.sub(r"TinyML Studio \d+\.\d+\.\d+\s*\|\s*User Manual|Page \d+", "", text).strip()
    images = len(page.images)
    if not body and not images:
        problems.append(f"page {i}: blank")
    elif len(body) < MIN_CHARS and not images:
        problems.append(f"page {i}: almost empty ({len(body)} characters, no image)")
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^[-–—•]\s", s):
            problems.append(f"page {i}: dash or bullet line: {s[:50]}")
    if "—" in text or "–" in text:
        problems.append(f"page {i}: contains a long dash")
    print(f"  page {i:>2}: {len(body):>5} chars, {images} image(s)")
print("problems:", len(problems))
for p in problems:
    print(" ", p)
sys.exit(1 if problems else 0)
