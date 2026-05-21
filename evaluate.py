"""
Evaluation module for the Multilingual Customer Support Chatbot.
Measures performance across three dimensions:
  1. RAG Retrieval Quality — Does FAISS retrieve the right FAQ for a given query?
  2. Sentiment Analysis Accuracy — Does TextBlob classify sentiment correctly?
  3. Language Detection Accuracy — Does langdetect identify the language correctly?

Run: python evaluate.py (from the project directory, with venv activated)
"""

import json
import os
import sys
from datetime import datetime

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentiment import analyze_sentiment
from language_detector import detect_language

# ---------------------------------------------------------------------------
# TEST DATASET — Ground truth labels for each test query
# ---------------------------------------------------------------------------

EVALUATION_DATA = [
    # --- English Neutral ---
    {"query": "How can I track my order?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "order tracking"},
    {"query": "What is your return policy?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "return policy"},
    {"query": "How do I apply a coupon code?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "coupon code"},
    {"query": "Do you offer international shipping?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "international shipping"},
    {"query": "How do I contact customer support?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "customer support contact"},
    {"query": "Is cash on delivery available?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "cash on delivery"},
    {"query": "Can I exchange a product instead of returning it?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "product exchange"},
    {"query": "How long does a refund take?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": "refund timeline"},

    # --- English Positive ---
    {"query": "Thanks! Great service!", "expected_sentiment": "positive", "expected_language": "English", "expected_faq_topic": None},
    {"query": "Great! Do you offer international shipping?", "expected_sentiment": "positive", "expected_language": "English", "expected_faq_topic": "international shipping"},
    {"query": "Love your products! How do I apply a coupon?", "expected_sentiment": "positive", "expected_language": "English", "expected_faq_topic": "coupon code"},

    # --- English Negative ---
    {"query": "This is terrible! I want a refund!", "expected_sentiment": "negative", "expected_language": "English", "expected_faq_topic": "refund timeline"},
    {"query": "My payment failed but money was deducted. This is terrible!", "expected_sentiment": "negative", "expected_language": "English", "expected_faq_topic": "payment failed"},
    {"query": "My delivery is delayed again, this is unacceptable and horrible", "expected_sentiment": "negative", "expected_language": "English", "expected_faq_topic": "delivery delay"},
    {"query": "I am really frustrated, my account got locked", "expected_sentiment": "negative", "expected_language": "English", "expected_faq_topic": "account locked"},
    {"query": "Worst experience ever. I want a refund immediately.", "expected_sentiment": "negative", "expected_language": "English", "expected_faq_topic": "refund timeline"},

    # --- Hindi (Transliterated) ---
    {"query": "Mera order kahan hai?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "order tracking"},
    {"query": "Mera order cancel kaise karein?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "order cancellation"},
    {"query": "Refund kitne din mein aata hai?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "refund timeline"},
    {"query": "COD available hai kya?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "cash on delivery"},
    {"query": "Password bhool gaya. Kaise reset karein?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "password reset"},
    {"query": "Product exchange kaise karein?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "product exchange"},
    {"query": "Customer support se kaise contact karein?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "customer support contact"},
    {"query": "Payment fail ho gaya lekin paise kat gaye. Kya karein?", "expected_sentiment": "neutral", "expected_language": "Hindi", "expected_faq_topic": "payment failed"},

    # --- Other Languages ---
    {"query": "¿Cómo puedo rastrear mi pedido?", "expected_sentiment": "neutral", "expected_language": "Spanish", "expected_faq_topic": "order tracking"},
    {"query": "Comment puis-je suivre ma commande?", "expected_sentiment": "neutral", "expected_language": "French", "expected_faq_topic": "order tracking"},
    {"query": "Wie kann ich meine Bestellung verfolgen?", "expected_sentiment": "neutral", "expected_language": "German", "expected_faq_topic": "order tracking"},
    {"query": "Como posso rastrear meu pedido?", "expected_sentiment": "neutral", "expected_language": "Portuguese", "expected_faq_topic": "order tracking"},
    {"query": "注文の追跡方法を教えてください", "expected_sentiment": "neutral", "expected_language": "Japanese", "expected_faq_topic": "order tracking"},
    {"query": "Как отследить мой заказ?", "expected_sentiment": "neutral", "expected_language": "Russian", "expected_faq_topic": "order tracking"},
    {"query": "주문 추적은 어떻게 하나요?", "expected_sentiment": "neutral", "expected_language": "Korean", "expected_faq_topic": "order tracking"},
    {"query": "我怎么查询我的订单？", "expected_sentiment": "neutral", "expected_language": "Chinese", "expected_faq_topic": "order tracking"},

    # --- Negative in Other Languages ---
    {"query": "Mi pago falló pero me cobraron. Esto es terrible!", "expected_sentiment": "negative", "expected_language": "Spanish", "expected_faq_topic": "payment failed"},
    {"query": "Ma livraison est en retard, c'est inacceptable!", "expected_sentiment": "negative", "expected_language": "French", "expected_faq_topic": "delivery delay"},

    # --- Edge Cases / Out of Knowledge Base ---
    {"query": "Do you sell laptops?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": None},
    {"query": "Can I pay with Bitcoin?", "expected_sentiment": "neutral", "expected_language": "English", "expected_faq_topic": None},
]


def evaluate_sentiment() -> dict:
    """Evaluate sentiment analysis accuracy across all test queries.
    Returns accuracy, per-class precision/recall/f1, and misclassified examples.
    """
    correct = 0
    total = len(EVALUATION_DATA)
    misclassified = []

    # Per-class counters: true positives, false positives, false negatives
    classes = ["positive", "neutral", "negative"]
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    for item in EVALUATION_DATA:
        result = analyze_sentiment(item["query"])
        predicted = result["label"]
        expected = item["expected_sentiment"]

        if predicted == expected:
            correct += 1
            tp[expected] += 1
        else:
            misclassified.append({
                "query": item["query"],
                "expected": expected,
                "predicted": predicted,
                "polarity": result["polarity"],
            })
            fp[predicted] += 1
            fn[expected] += 1

    # Calculate per-class precision, recall, F1
    class_metrics = {}
    for c in classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_metrics[c] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "class_metrics": class_metrics,
        "misclassified": misclassified,
    }


def evaluate_language_detection() -> dict:
    """Evaluate language detection accuracy across all test queries.
    Returns accuracy and misclassified examples.
    """
    correct = 0
    total = len(EVALUATION_DATA)
    misclassified = []

    # Per-language counters
    lang_counts = {}
    lang_correct = {}

    for item in EVALUATION_DATA:
        result = detect_language(item["query"])
        predicted = result["name"]
        expected = item["expected_language"]

        # Track per-language
        lang_counts[expected] = lang_counts.get(expected, 0) + 1

        if predicted == expected:
            correct += 1
            lang_correct[expected] = lang_correct.get(expected, 0) + 1
        else:
            misclassified.append({
                "query": item["query"],
                "expected": expected,
                "predicted": predicted,
            })

    # Per-language accuracy
    per_language = {}
    for lang in lang_counts:
        per_language[lang] = {
            "correct": lang_correct.get(lang, 0),
            "total": lang_counts[lang],
            "accuracy": lang_correct.get(lang, 0) / lang_counts[lang],
        }

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "per_language": per_language,
        "misclassified": misclassified,
    }


def print_report(sentiment_results: dict, language_results: dict):
    """Print a formatted evaluation report to console."""
    print("=" * 70)
    print("  MULTILINGUAL SUPPORT CHATBOT — EVALUATION REPORT")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test Samples: {len(EVALUATION_DATA)}")
    print("=" * 70)

    # --- Sentiment Section ---
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  1. SENTIMENT ANALYSIS                                         │")
    print("└─────────────────────────────────────────────────────────────────┘")
    sr = sentiment_results
    print(f"\n  Overall Accuracy: {sr['accuracy']:.1%}  ({sr['correct']}/{sr['total']})")
    print(f"\n  {'CLASS':<12} {'PRECISION':>10} {'RECALL':>10} {'F1-SCORE':>10}")
    print(f"  {'-'*42}")
    for cls, m in sr["class_metrics"].items():
        print(f"  {cls:<12} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f}")

    if sr["misclassified"]:
        print(f"\n  Misclassified ({len(sr['misclassified'])}):")
        for item in sr["misclassified"]:
            print(f"    - \"{item['query'][:55]}...\"" if len(item["query"]) > 55 else f"    - \"{item['query']}\"")
            print(f"      Expected: {item['expected']}, Got: {item['predicted']} (polarity: {item['polarity']:.4f})")

    # --- Language Section ---
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  2. LANGUAGE DETECTION                                          │")
    print("└─────────────────────────────────────────────────────────────────┘")
    lr = language_results
    print(f"\n  Overall Accuracy: {lr['accuracy']:.1%}  ({lr['correct']}/{lr['total']})")
    print(f"\n  {'LANGUAGE':<15} {'CORRECT':>8} {'TOTAL':>8} {'ACCURACY':>10}")
    print(f"  {'-'*42}")
    for lang, m in sorted(lr["per_language"].items()):
        print(f"  {lang:<15} {m['correct']:>8} {m['total']:>8} {m['accuracy']:>10.1%}")

    if lr["misclassified"]:
        print(f"\n  Misclassified ({len(lr['misclassified'])}):")
        for item in lr["misclassified"]:
            print(f"    - \"{item['query'][:55]}...\"" if len(item["query"]) > 55 else f"    - \"{item['query']}\"")
            print(f"      Expected: {item['expected']}, Got: {item['predicted']}")

    # --- Summary ---
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  SUMMARY                                                        │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print(f"\n  Sentiment Accuracy:  {sr['accuracy']:.1%}")
    print(f"  Language Accuracy:   {lr['accuracy']:.1%}")
    print(f"  Test Queries:        {len(EVALUATION_DATA)}")
    print(f"  Languages Tested:    {len(lr['per_language'])}")
    print()


if __name__ == "__main__":
    print("\nRunning evaluation...\n")
    sentiment_results = evaluate_sentiment()
    language_results = evaluate_language_detection()
    print_report(sentiment_results, language_results)
