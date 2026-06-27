"""Text preprocessing for the Steam reviews NLP project.

Shared, model agnostic text cleaning.
variant specific decisions (stopwords, n-grams, lemmatization) live in the vectorizers of the modeling notebooks."""

import html

def clean_text(text: str) -> str:
    """Applay the shared cleaning steps to a single review string"""

    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text) # return html code
    text = text.lower() # lower case
    text = " ".join(text.split()) # clean whitespaces
    return text