import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# CONFIGURATION
# =========================

CLASS_DESCRIPTIONS = {
    "fallacy of credibility": (
        "Fallacy of credibility wrongly accepts or rejects a claim based only on "
        "who said it, not on actual evidence or reasoning. "
        "Example: 'This medicine must work because a famous actor endorses it.' "
        "Or: 'We should ignore that research because the scientist is young and inexperienced.'"
    ),
    "ad hominem": (
        "Ad hominem attacks the person making the argument instead of addressing "
        "the argument itself. It dismisses a claim by insulting or discrediting the person. "
        "Example: 'You cannot trust his opinion on climate change, he failed science in school.' "
        "Example: 'She is just a housewife, what would she know about economics?'"
    ),
    "faulty generalization": (
        "Faulty generalization draws a broad conclusion from too small or unrepresentative "
        "a sample. It assumes what is true for a few is true for all. "
        "Example: 'I met two dishonest lawyers, so all lawyers must be dishonest.' "
        "Example: 'My grandfather smoked and lived to 90, so smoking is not harmful.'"
    ),
    "false causality": (
        "False causality assumes that because one event followed another, the first "
        "event caused the second, without proper evidence of a causal link. "
        "Example: 'After the new mayor took office crime went up, so the mayor caused it.' "
        "Example: 'I wore my lucky socks and we won the match, so the socks made us win.'"
    ),
    "appeal to emotion": (
        "Appeal to emotion uses feelings like fear, pity, guilt or pride instead of "
        "logical reasoning to convince someone of a conclusion. "
        "Example: 'Think of the innocent children suffering. How can you not donate?' "
        "Example: 'Our soldiers died for this country. You must support this war policy.'"
    ),
}

NEWS_DOMAINS = [
    "political news reporting",
    "economic news reporting",
    "health and medical news",
    "environmental news",
    "technology news",
    "education policy news",
    "crime and justice reporting",
    "social issues reporting",
    "sports journalism",
    "science journalism",
]

SIMPLE_DOMAINS = [
    "politics",
    "health",
    "education",
    "economics",
    "everyday life",
    "environment",
]

# =========================
# GENERATION FUNCTIONS
# =========================

def generate_news_style(fallacy_name, description, domain, count):
    """Generate complex news-article style examples."""

    prompt = f"""You are an expert in logical fallacies and journalism.

Fallacy type: {fallacy_name}
Definition: {description}
Context: {domain}

Generate {count} examples of the {fallacy_name} fallacy written in the style of real news articles or opinion pieces.
The examples should:
- Sound like something written in a newspaper, blog, or news website
- Be subtle and indirect, not obvious textbook examples
- Be 2-4 sentences long, embedded in realistic reporting context
- Use varied sentence structures and vocabulary
- NOT start with obvious trigger phrases like "Either..." or "Everyone knows..."
- Feel natural, like real arguments people make in news commentary

Output ONLY the examples, one per line with a blank line between each.
No numbering, no labels, no explanations."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in logical fallacies and journalism who generates realistic news-style examples."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.95
        )

        response_text = completion.choices[0].message.content.strip()

        # Split by blank lines first, then by newlines
        chunks = [c.strip() for c in response_text.split('\n\n') if c.strip() and len(c.strip()) > 30]
        if len(chunks) < count:
            lines = [l.strip() for l in response_text.split('\n') if l.strip() and len(l.strip()) > 30]
            chunks = lines

        return chunks[:count]

    except Exception as e:
        print(f"    Error: {e}")
        time.sleep(3)
        return []


def generate_simple_style(fallacy_name, description, domain, count):
    """Generate clear pattern-based examples."""

    prompt = f"""You are an expert in logical fallacies.

Fallacy type: {fallacy_name}
Definition: {description}
Domain: {domain}

Generate {count} clear, direct examples of the {fallacy_name} fallacy in the context of {domain}.
The examples should:
- Clearly and obviously demonstrate the fallacy
- Be 1-2 sentences long
- Sound like something a real person would say in conversation or debate
- Be varied in wording and structure

Output ONLY the examples, one per line.
No numbering, no labels, no explanations."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in logical fallacies who generates clear realistic examples."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=600,
            temperature=0.85
        )

        response_text = completion.choices[0].message.content.strip()
        lines = [l.strip() for l in response_text.split('\n') if l.strip() and len(l.strip()) > 20]
        return lines[:count]

    except Exception as e:
        print(f"    Error: {e}")
        time.sleep(3)
        return []


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    all_rows = []

    print("=" * 55)
    print("Full batch augmentation — 80 examples per class")
    print("50 news-style + 30 simple per class")
    print("=" * 55)

    for fallacy, description in CLASS_DESCRIPTIONS.items():

        print(f"\n{'='*55}")
        print(f"Generating: {fallacy.upper()}")
        print(f"{'='*55}")

        fallacy_examples = []

        # --- NEWS STYLE: 5 examples per domain x 10 domains = 50 ---
        print("  Generating news-style examples...")
        for domain in NEWS_DOMAINS:
            print(f"    Domain: {domain}")
            examples = generate_news_style(fallacy, description, domain, 5)
            fallacy_examples.extend(examples)
            time.sleep(1.5)

        news_count = len(fallacy_examples)
        print(f"  News-style total: {news_count}")

        # --- SIMPLE STYLE: 5 examples per domain x 6 domains = 30 ---
        print("  Generating simple-style examples...")
        for domain in SIMPLE_DOMAINS:
            print(f"    Domain: {domain}")
            examples = generate_simple_style(fallacy, description, domain, 5)
            fallacy_examples.extend(examples)
            time.sleep(1.5)

        total_count = len(fallacy_examples)
        print(f"  Simple-style total: {total_count - news_count}")
        print(f"  Total for {fallacy}: {total_count}")

        for example in fallacy_examples:
            all_rows.append({
                "source_article": example,
                "updated_label": fallacy
            })

    # Save augmented data
    output_path = os.path.join(os.path.dirname(__file__), "raw", "augmented_full_batch.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 55)
    print(f"Done! Total examples generated: {len(all_rows)}")
    print(f"Saved to: {output_path}")
    print("=" * 55)

    print("\nFinal count per class:")
    print(df['updated_label'].value_counts())

    print("\nPREVIEW - 2 news-style + 2 simple examples per class:")
    print("-" * 55)
    for fallacy in CLASS_DESCRIPTIONS.keys():
        subset = df[df['updated_label'] == fallacy]
        print(f"\n{fallacy.upper()}:")
        for _, row in subset.head(2).iterrows():
            print(f"  [news]   -> {row['source_article'][:120]}")
        for _, row in subset.tail(2).iterrows():
            print(f"  [simple] -> {row['source_article'][:120]}")