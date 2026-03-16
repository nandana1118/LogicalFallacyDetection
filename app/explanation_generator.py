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
            max_tokens=150
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:

        print("Groq explanation error:", e)
        return "Explanation unavailable."


def verify_fallacy(text, label):

    prompt = f"""
Does the following text actually contain a {label} fallacy?

Text: {text}

Reply with only YES or NO.
"""

    try:

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert in logical reasoning and fallacy detection. Reply only with YES or NO."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5
        )

        response = completion.choices[0].message.content.strip().upper()
        return response.startswith("YES")

    except Exception as e:

        print("Groq verification error:", e)
        return True  # if API fails, default to accepting BERT's prediction