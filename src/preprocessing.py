"""Text preprocessing for the Steam reviews NLP project.

Shared, model agnostic text cleaning.
variant specific decisions (stopwords, n-grams, lemmatization) live in the vectorizers of the modeling notebooks."""

import html

def clean_text(text: str) -> str:
    """Apply the shared cleaning steps to a single review string"""

    if not isinstance(text, str):
        return ""

    # html.unescape follows HTML5 rules, so broken entities without a trailing
    # semicolon ('&gt') are decoded too — no extra regex needed.
    text = html.unescape(text) # decode html entities
    text = text.lower() # lower case
    text = " ".join(text.split()) # clean whitespaces
    return text