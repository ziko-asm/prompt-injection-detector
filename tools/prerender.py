"""Bake the results-driven parts of index.html into the HTML itself.

The page draws its tiles, charts and tables from results/results.json in app.js.
Shipping that output inside index.html means it is visible at first paint (no
layout shift, faster LCP, numbers readable by search engines). app.js still
re-renders from results.json on load, which attaches tooltips and filters.

Run after results/results.json changes:

    python tools/prerender.py

Needs Chrome or Edge installed. Serves the repo on a local port, loads the page
headless, then copies each container's rendered HTML back into index.html.
"""

import functools
import http.server
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.html"
IDS = ["tiles", "find", "gapChart", "costChart", "cascadeTbl", "cascadeNote",
       "filters", "allTbl", "dataList", "gen"]
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def find_browser():
    for b in BROWSERS:
        if os.path.exists(b) or shutil.which(b):
            return b
    raise SystemExit("Chrome or Edge not found")


def inner_span(html, el_id):
    """(start, end) of the inner HTML of the element with this id."""
    m = re.search(r'<(\w+)[^>]*\bid="' + re.escape(el_id) + r'"[^>]*>', html)
    if not m:
        raise SystemExit(f"#{el_id} not found")
    tag, start, depth = m.group(1), m.end(), 1
    for t in re.finditer(r"<(/?)" + tag + r"\b[^>]*?(/?)>", html[start:]):
        if t.group(2):
            continue
        depth += -1 if t.group(1) else 1
        if depth == 0:
            return start, start + t.start()
    raise SystemExit(f"#{el_id} is not closed")


def dump_dom(browser, url):
    out = subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-first-run",
         "--virtual-time-budget=10000", "--window-size=1350,900", "--dump-dom", url],
        check=False, capture_output=True, text=True, encoding="utf-8", timeout=120)
    if "<html" not in out.stdout:
        raise SystemExit("browser returned no DOM:\n" + out.stderr[-2000:])
    return out.stdout


def main():
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rendered = dump_dom(find_browser(), f"http://127.0.0.1:{server.server_port}/index.html")
    finally:
        server.shutdown()

    src = INDEX.read_bytes().decode("utf-8")
    for el_id in IDS:
        s, e = inner_span(rendered, el_id)
        # index.html uses CRLF line endings; keep them consistent.
        snippet = rendered[s:e].replace("\r\n", "\n").replace("\n", "\r\n")
        if not snippet.strip():
            raise SystemExit(f"#{el_id} rendered empty; is results/results.json valid?")
        s, e = inner_span(src, el_id)
        src = src[:s] + snippet + src[e:]
    INDEX.write_bytes(src.encode("utf-8"))
    print(f"Pre-rendered {len(IDS)} blocks into {INDEX.name}")


if __name__ == "__main__":
    main()
