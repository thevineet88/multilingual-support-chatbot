import streamlit as st

from rag_engine import get_rag_response
from sentiment import analyze_sentiment, get_empathy_message
from language_detector import detect_language

# --- Page config ---
st.set_page_config(
    page_title="Multilingual Support Chatbot",
    page_icon="💬",
    layout="wide",
)

# --- Custom CSS for professional look ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .user-msg {
        background-color: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 6px 0;
        max-width: 75%;
        float: right;
        clear: both;
        word-wrap: break-word;
    }
    .bot-msg {
        background-color: #ffffff;
        color: #1a1a2e;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 4px;
        margin: 6px 0;
        max-width: 75%;
        float: left;
        clear: both;
        border: 1px solid #e0e0e0;
        word-wrap: break-word;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        margin-right: 6px;
        color: white;
    }
    .badge-green  { background-color: #28a745; }
    .badge-orange { background-color: #fd7e14; }
    .badge-red    { background-color: #dc3545; }
    .badge-lang   { background-color: #6f42c1; }
    .chat-container {
        overflow-y: auto;
        padding: 10px;
    }
    .clearfix::after {
        content: "";
        display: table;
        clear: both;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] li,
    div[data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0
if "languages_seen" not in st.session_state:
    st.session_state.languages_seen = set()
if "total_response_chars" not in st.session_state:
    st.session_state.total_response_chars = 0

# --- Sidebar ---
with st.sidebar:
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        An **AI-powered customer support chatbot** built for
        e-commerce businesses that serve a global, multilingual
        customer base.

        **Why this exists:** Traditional support bots fail when
        customers write in languages other than English, miss
        emotional cues in frustrated messages, and hallucinate
        answers that don't match actual store policies.

        This chatbot solves all three — it detects the customer's
        language automatically, senses frustration and responds
        with empathy, and grounds every answer in a verified
        FAQ knowledge base so it never makes up information.
        """
    )

    st.markdown("---")
    st.subheader("📊 Performance Metrics")

    total = st.session_state.total_queries
    neg = st.session_state.negative_count
    avg_len = (
        round(st.session_state.total_response_chars / total) if total > 0 else 0
    )
    langs = sorted(st.session_state.languages_seen) if st.session_state.languages_seen else ["—"]

    col1, col2 = st.columns(2)
    col1.metric("Total Queries", total)
    col2.metric("Neg. Sentiment %", f"{round(neg / total * 100) if total else 0}%")

    col3, col4 = st.columns(2)
    col3.metric("Avg Response Len", f"{avg_len} chars")
    col4.metric("Unique Languages", len(st.session_state.languages_seen))

    st.caption(f"Languages detected: {', '.join(langs)}")

# --- Main chat area ---
st.title("💬 Multilingual Customer Support")
st.caption("Ask anything about orders, returns, payments, or account issues — in any language.")

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="clearfix"><div class="user-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        # Build badge HTML
        s = msg.get("sentiment", {})
        l = msg.get("language", {})
        sentiment_cls = f"badge-{s.get('color', 'orange')}"
        badges = (
            f'<span class="badge {sentiment_cls}">{s.get("label", "neutral")}</span>'
            f'<span class="badge badge-lang">{l.get("name", "—")}</span>'
        )
        st.markdown(
            f'<div class="clearfix"><div class="bot-msg">{badges}<br/>{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Analyse input
    sentiment = analyze_sentiment(user_input)
    language = detect_language(user_input)

    # Update metrics
    st.session_state.total_queries += 1
    if sentiment["label"] == "negative":
        st.session_state.negative_count += 1
    st.session_state.languages_seen.add(language["name"])

    # Get RAG response
    empathy = get_empathy_message(sentiment)
    answer = get_rag_response(user_input)
    full_response = empathy + answer

    st.session_state.total_response_chars += len(full_response)

    # Store bot message with metadata
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sentiment": sentiment,
            "language": language,
        }
    )

    # Rerun to display new messages
    st.rerun()
