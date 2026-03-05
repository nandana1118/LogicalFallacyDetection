import requests
from bs4 import BeautifulSoup


def extract_article_text(url):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print("Could not download page.")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")

        article_text = " ".join([p.get_text() for p in paragraphs])

        if article_text.strip() == "":
            return None

        return article_text

    except Exception as e:
        print("Extraction error:", e)
        return None