import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

BASE = "https://docs.python.org/3/tutorial/"
RAW_DIR = Path("data/raw")
OUT = Path("data/chunks.jsonl")


def page_url(name):
    return BASE + name


def clean_title(text):
    text = text.replace("¶", "").strip()
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text)
    return " ".join(text.split())


def clean_text(node):
    clone = BeautifulSoup(str(node), "lxml")
    root = clone.body or clone

    for sel in [
        "div.related",
        "div.sphinxsidebar",
        "div.footer",
        "a.headerlink",
        "script",
        "style",
    ]:
        for x in root.select(sel):
            x.decompose()

    return " ".join(root.get_text(" ").split())


def section_text(section):
    parts = []

    for child in section.children:
        if not isinstance(child, Tag):
            continue

        if child.name == "section":
            continue

        if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            continue

        text = clean_text(child)
        if text:
            parts.append(text)

    return " ".join(parts)


def chunks_from_file(path):
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    body = soup.find("div", {"class": "body"}) or soup
    url = page_url(path.name)
    title = clean_title(
        soup.title.get_text(" ", strip=True) if soup.title else path.stem
    )

    result = []
    sections = body.find_all("section")

    for section in sections:
        heading = section.find(
            ["h1", "h2", "h3", "h4", "h5", "h6"],
            recursive=False,
        )

        if heading is None:
            continue

        anchor = section.get("id") or heading.get("id") or ""
        section_title = clean_title(heading.get_text(" ", strip=True))
        text = section_text(section)

        if len(text) < 40:
            continue

        url_with_anchor = url + (("#" + anchor) if anchor else "")

        result.append({
            "text": text,
            "url": url_with_anchor,
            "anchor": anchor,
            "section_title": section_title,
            "page_title": title,
        })

    return result


def main():
    all_chunks = []

    for path in sorted(RAW_DIR.glob("*.html")):
        chunks = chunks_from_file(path)
        all_chunks.extend(chunks)
        print(path.name, len(chunks))

    with OUT.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            chunk["id"] = i
            f.write(json.dumps(chunk, ensure_ascii=False) + chr(10))

    print("Total chunks:", len(all_chunks))


if __name__ == "__main__":
    main()
