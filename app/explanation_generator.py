import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_explanation(text, label):

    prompt = f"""
The following text contains a {label} logical fallacy.

Text: {text}

Explain briefly in 2 sentences why this reasoning is flawed.
"""

    try:

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert in logical reasoning and fallacy detection."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:

        print("Groq explanation error:", e)
        return "Explanation unavailable."