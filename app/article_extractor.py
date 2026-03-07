import requests
from bs4 import BeautifulSoup


def extract_article_text(url):

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print("Could not download page.")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")

        cleaned_paragraphs = []

        blacklist = [
            "newsletter",
            "subscribe",
            "email address",
            "all rights reserved",
            "copyright",
            "sign up",
            "comment",
            "follow us",
            "join here"
        ]

        for p in paragraphs:

            text = p.get_text().strip()

            # skip very short lines
            if len(text) < 60:
                continue

            # skip boilerplate phrases
            if any(word in text.lower() for word in blacklist):
                continue

            cleaned_paragraphs.append(text)

        # limit to first 12 paragraphs to avoid footer junk
        article_text = " ".join(cleaned_paragraphs[:12])

        return article_text if article_text else None

    except Exception as e:
        print("Extraction error:", e)
        return None