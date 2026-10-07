import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUTPUT = Path("data/raw/documents.jsonl")

SOURCES = [
    {
        "document_id": "neu-grad-policies",
        "title": "Graduate Academic Policies and Procedures",
        "url": "https://catalog.northeastern.edu/graduate/academic-policies-procedures/",
        "section": "Graduate Academic Policies",
        "document_type": "policy",
    },
    {
        "document_id": "neu-drop-class",
        "title": "Dropping a Class",
        "url": "https://catalog.northeastern.edu/graduate/academic-policies-procedures/drop-class/",
        "section": "Graduate Academic Policies",
        "document_type": "policy",
    },
    {
        "document_id": "neu-academic-calendar",
        "title": "Academic Calendar",
        "url": "https://catalog.northeastern.edu/graduate/academic-policies-procedures/academic-calendar/",
        "section": "Graduate Academic Policies",
        "document_type": "calendar",
    },
    {
        "document_id": "neu-coop",
        "title": "Graduate Cooperative Education",
        "url": "https://catalog.northeastern.edu/graduate/academic-policies-procedures/cooperative-education/",
        "section": "Experiential Learning",
        "document_type": "policy",
    },
    {
        "document_id": "neu-cpt",
        "title": "F-1 Curricular Practical Training",
        "url": "https://international.northeastern.edu/ogs/employment/off-campus-employment/f-1-curricular-practical-training/",
        "section": "International Student Employment",
        "document_type": "policy",
    },
]


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that usually contain navigation/noise
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]
    ):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find(id="content")
        or soup.find(class_="page-content")
        or soup.find(class_="content")
        or soup.body
    )

    if main is None:
        return ""

    text = main.get_text(" ", strip=True)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def fetch_page(src: dict) -> str | None:
    print(f"Fetching: {src['title']}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
    }

    try:
        response = requests.get(
            src["url"],
            timeout=30,
            headers=headers,
        )

        if response.status_code == 403:
            print(f"Skipped (403 Forbidden): {src['title']}")
            return None

        response.raise_for_status()

        text = clean_text(response.text)

        if not text or len(text) < 100:
            print(f"Skipped (empty or too short): {src['title']}")
            return None

        print(f"  Collected {len(text):,} characters")
        return text

    except requests.RequestException as exc:
        print(f"Skipped due to request error: {src['title']}")
        print(f"  {exc}")
        return None


def collect():
    rows = []

    for src in SOURCES:
        text = fetch_page(src)

        if text is None:
            continue

        row = {
            "document_id": src["document_id"],
            "title": src["title"],
            "source_url": src["url"],
            "section": src["section"],
            "updated_at": "2026-2027",
            "document_type": src["document_type"],
            "text": text,
        }

        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Saved {len(rows)} documents to {OUTPUT}")

    if len(rows) < len(SOURCES):
        print(
            f"Skipped {len(SOURCES) - len(rows)} source(s). "
            "These can be added manually later if necessary."
        )


if __name__ == "__main__":
    collect()