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

# Slightly more for ad populum since it was borderline in testing
SAMPLES_CONFIG = {
    "ad populum":           {"news": 40, "simple": 20},
    "false dilemma":        {"news": 35, "simple": 15},
    "circular reasoning":   {"news": 35, "simple": 15},
    "fallacy of logic":     {"news": 35, "simple": 15},
    "fallacy of relevance": {"news": 35, "simple": 15},
    "fallacy of extension": {"news": 35, "simple": 15},
    "intentional":          {"news": 35, "simple": 15},
}

CLASS_DESCRIPTIONS = {
    "ad populum": (
        "Ad populum (appeal to popularity) argues that something is true or correct "
        "simply because many people believe it or do it. "
        "Example: 'Millions of people use this supplement so it must be effective.' "
        "Example: 'Most voters support this policy so it must be the right decision.'"
    ),
    "false dilemma": (
        "False dilemma presents only two options as if they are the only possibilities, "
        "when in reality there are other alternatives. "
        "Example: 'You are either with us or against us.' "
        "Example: 'We must cut education funding or raise taxes — there is no other way.'"
    ),
    "circular reasoning": (
        "Circular reasoning uses the conclusion as a premise to support itself, "
        "going in a loop without providing real evidence. "
        "Example: 'The Bible is true because it says so in the Bible.' "
        "Example: 'This policy works because it is an effective policy.'"
    ),
    "fallacy of logic": (
        "Fallacy of logic involves a general error in the logical structure of an argument "
        "where the conclusion does not follow from the premises. "
        "Example: 'All cats have fur. This animal has fur. Therefore it must be a cat.' "
        "Example: 'If it rains the ground gets wet. The ground is wet, so it must have rained.'"
    ),
    "fallacy of relevance": (
        "Fallacy of relevance uses information or evidence that is not actually relevant "
        "to the conclusion being drawn, distracting from the real argument. "
        "Example: 'We should not listen to his views on taxation because he drives an expensive car.' "
        "Example: 'She supports animal rights but she wears leather shoes, so her argument is invalid.'"
    ),
    "fallacy of extension": (
        "Fallacy of extension misrepresents or exaggerates someone's argument to make "
        "it easier to attack, also known as straw man. "
        "Example: 'She said we should eat less meat, so she wants everyone to starve.' "
        "Example: 'He wants stricter gun laws, which means he wants to take away all freedom.'"
    ),
    "intentional": (
        "Intentional fallacy involves deliberately using misleading reasoning or "
        "manipulative rhetoric knowing it is logically flawed, to deceive or manipulate. "
        "Example: A politician deliberately misquotes statistics to mislead voters. "
        "Example: An advertiser knowingly uses false comparisons to make their product seem superior."
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
]

SIMPLE_DOMAINS = [
    "politics",
    "health",
    "education",
    "economics",
    "everyday life",
]

# =========================
# GENERATION FUNCTIONS
# =========================

def generate_news_style(fallacy_name, description, domain, count):
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
    print("Remaining classes augmentation")
    print("=" * 55)

    for fallacy, config in SAMPLES_CONFIG.items():

        description = CLASS_DESCRIPTIONS[fallacy]
        news_target = config["news"]
        simple_target = config["simple"]

        print(f"\n{'='*55}")
        print(f"Generating: {fallacy.upper()}")
        print(f"Target: {news_target} news-style + {simple_target} simple")
        print(f"{'='*55}")

        fallacy_examples = []

        # --- NEWS STYLE ---
        print("  Generating news-style examples...")
        per_domain = max(1, news_target // len(NEWS_DOMAINS))
        remaining = news_target

        for domain in NEWS_DOMAINS:
            if remaining <= 0:
                break
            count = min(per_domain, remaining)
            print(f"    Domain: {domain}")
            examples = generate_news_style(fallacy, description, domain, count)
            fallacy_examples.extend(examples)
            remaining -= len(examples)
            time.sleep(1.5)

        news_count = len(fallacy_examples)
        print(f"  News-style total: {news_count}")

        # --- SIMPLE STYLE ---
        print("  Generating simple-style examples...")
        per_domain = max(1, simple_target // len(SIMPLE_DOMAINS))
        remaining = simple_target

        for domain in SIMPLE_DOMAINS:
            if remaining <= 0:
                break
            count = min(per_domain, remaining)
            print(f"    Domain: {domain}")
            examples = generate_simple_style(fallacy, description, domain, count)
            fallacy_examples.extend(examples)
            remaining -= len(examples)
            time.sleep(1.5)

        total_count = len(fallacy_examples)
        print(f"  Simple-style total: {total_count - news_count}")
        print(f"  Total for {fallacy}: {total_count}")

        for example in fallacy_examples:
            all_rows.append({
                "source_article": example,
                "updated_label": fallacy
            })

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "raw", "augmented_remaining_batch.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 55)
    print(f"Done! Total examples generated: {len(all_rows)}")
    print(f"Saved to: {output_path}")
    print("=" * 55)

    print("\nFinal count per class:")
    print(df['updated_label'].value_counts())

    print("\nPREVIEW - 2 examples per class:")
    print("-" * 55)
    for fallacy in SAMPLES_CONFIG.keys():
        subset = df[df['updated_label'] == fallacy]
        print(f"\n{fallacy.upper()}:")
        for _, row in subset.head(2).iterrows():
            print(f"  [news]   -> {row['source_article'][:120]}")
        for _, row in subset.tail(2).iterrows():
            print(f"  [simple] -> {row['source_article'][:120]}")
