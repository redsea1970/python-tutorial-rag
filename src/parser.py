import json
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE = "https://docs.python.org/3/tutorial/"
START = "index.html"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-rag)"}

def fetch(client, url):
    r = client.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r.text

def save_raw(url, html):
    name = url.rstrip("/").split("/")[-1] or "index.html"
    if not name.endswith(".html"):
        name = name + ".html"
    (RAW_DIR / name).write_text(html, encoding="utf-8")

def main():
    visited = set()
    queue = [urljoin(BASE, START)]
    client = httpx.Client(headers=HEADERS)
    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            html = fetch(client, url)
        except Exception as e:
            print("SKIP", url, e)
            continue
        save_raw(url, html)
        print("OK", url)
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("div.related a, div.sphinxsidebar a, a.reference.internal"):
            href = a.get("href")
            if not href or href.startswith("#") or "://" in href:
                continue
            nxt = urljoin(url, href).split("#")[0]
            if nxt.startswith(BASE) and nxt not in visited:
                queue.append(nxt)
        time.sleep(0.2)
    print("Total pages:", len(visited))

if __name__ == "__main__":
    main()
