from langdetect import detect, DetectorFactory

# Fix seed for consistent language detection results
DetectorFactory.seed = 0

# Mapping of ISO 639-1 codes to display names (all languages langdetect supports)
LANGUAGE_MAP = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
}


# Common transliterated Hindi words that langdetect misclassifies as Indonesian/Malay.
# These words appear frequently in Indian customer support queries written in Roman script.
HINDI_KEYWORDS = {
    "mera", "meri", "mere", "kya", "hai", "hain", "kaise", "kahan", "kab",
    "kyun", "kyu", "karein", "karna", "karo", "kar", "chahiye", "chahte",
    "nahi", "nahin", "nhi", "aur", "ya", "bhi", "toh", "tho", "lekin",
    "agar", "abhi", "yahan", "wahan", "kitne", "kitna", "kitni", "kaun",
    "kaunsa", "humse", "humara", "aap", "aapka", "aapke", "aapki",
    "mujhe", "hum", "unka", "uska", "uski", "bhool", "bhul", "gaya",
    "gayi", "gaye", "hua", "hui", "hue", "hoga", "hogi", "raha", "rahi",
    "wala", "wali", "wale", "paise", "paisa", "rupee", "rupaye",
    "khareedna", "kharid", "order", "cancel", "refund", "return",
    "delivery", "dein", "dijiye", "batao", "bataye", "bataiye",
    "milega", "milegi", "milta", "sakta", "sakti", "sakte",
    "ghante", "din", "zyada", "kam", "bahut", "bohot", "achha", "accha",
    "bura", "kharab", "theek", "thik", "pehle", "baad", "andar",
    "available", "liye", "ke", "ki", "ka", "se", "mein", "par", "pe",
    "ko", "ne", "tak", "jaise", "jaisa", "karke", "hokar",
}


def _is_transliterated_hindi(text: str) -> bool:
    """Check if text contains enough Hindi keywords to override langdetect.
    Uses a simple word-overlap heuristic: if 30%+ of words are known Hindi
    words, classify as Hindi regardless of what langdetect says.
    """
    words = text.lower().split()
    if len(words) == 0:
        return False
    hindi_count = sum(1 for w in words if w.strip("?,!.") in HINDI_KEYWORDS)
    return (hindi_count / len(words)) >= 0.3


def detect_language(text: str) -> dict:
    """Detect the language of the input text.
    First checks for transliterated Hindi (Roman script Hindi that langdetect
    misclassifies as Indonesian). Falls back to langdetect for all other languages.
    Returns a dict with 'code' (ISO 639-1) and 'name' (human-readable).
    """
    try:
        # Check for transliterated Hindi first — langdetect can't handle it
        if _is_transliterated_hindi(text):
            return {"code": "hi", "name": "Hindi"}

        code = detect(text)
        name = LANGUAGE_MAP.get(code, code.upper())
        return {"code": code, "name": name}
    except Exception:
        return {"code": "en", "name": "English"}
