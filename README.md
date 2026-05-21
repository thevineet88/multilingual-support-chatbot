# Multilingual Customer Support Chatbot

A RAG-powered customer support chatbot that understands English and Hindi, detects user sentiment, and provides grounded answers from an e-commerce FAQ knowledge base using GPT-3.5-turbo.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TextBlob corpora
python -m textblob.download_corpora

# 4. Create a .env file with your OpenAI key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## How to Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Type a question in English or Hindi to get started.

## Screenshots

*Add screenshots of the running application here.*

## GenAI Techniques Used

1. **RAG (Retrieval-Augmented Generation)** — FAQ entries are embedded with OpenAI ada-002 and stored in a FAISS vector index; the top-3 matches are retrieved and passed as context to GPT-3.5-turbo so answers stay grounded in the knowledge base.
2. **Sentiment Analysis** — Each user message is scored with TextBlob; negative sentiment triggers an empathy-first response to acknowledge customer frustration.
3. **Language Detection** — The `langdetect` library identifies the input language so the bot can reply in the same language the customer used.
