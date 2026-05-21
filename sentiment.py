from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of user input using TextBlob.
    Returns a dict with 'label' (positive/neutral/negative),
    'polarity' (float -1 to 1), and 'color' (for UI badge).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        label, color = "positive", "green"
    elif polarity < -0.1:
        label, color = "negative", "red"
    else:
        label, color = "neutral", "orange"

    return {"label": label, "polarity": polarity, "color": color}


def get_empathy_message(sentiment_result: dict) -> str:
    """Return an empathy prefix if the user seems frustrated."""
    if sentiment_result["label"] == "negative":
        return "I can sense your frustration. Let me prioritize your issue.\n\n"
    return ""
