# Armenian Bank Voice AI Agent

A real-time, LLM-powered voice assistant designed for Armenian banks. Built on top of the [LiveKit Agents SDK](https://github.com/livekit/agents), this agent understands and speaks Armenian, providing accurate information about banking products and services using a Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Overview

The **Armenian Bank Voice AI** is a specialized customer support agent that handles queries related to:
*   **Credits**: Loan terms, interest rates, and eligibility.
*   **Deposits**: Savings accounts, term deposits, and interest conditions.
*   **Branch Locations**: Finding nearby branches and ATMs.

It utilizes a modern AI stack optimized for the Armenian language, ensuring high-quality transcriptions and natural-sounding speech.

## 🛠 Tech Stack

*   **Framework**: [LiveKit Agents](https://livekit.io/agents) (Python)
*   **STT (Speech-to-Text)**: Groq Whisper Large v3 (configured for Armenian)
*   **LLM (Language Model)**: Google Gemini 1.5 Flash (via Google AI Studio)
*   **TTS (Text-to-Speech)**: Google Gemini TTS (Armenian voice "Kore")
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Local/Embedded)
*   **RAG Engine**: [LlamaIndex](https://www.llamaindex.ai/)
*   **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Multilingual support)
*   **Package Manager**: [uv](https://github.com/astral-sh/uv)

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have `uv` installed on your system.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Installation
Clone the repository and install dependencies using `uv`:
```bash
git clone https://github.com/your-repo/armenian-bank-voice-ai.git
cd armenian-bank-voice-ai
uv sync
```

### 3. Environment Setup
Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```
Key requirements:
*   `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (from LiveKit Cloud or self-hosted)
*   `GOOGLE_API_KEY` (for Gemini LLM/TTS)
*   `GROQ_API_KEY` (for Whisper STT)

### 4. Data Ingestion
Populate the vector store with bank-specific data:
```bash
uv run python scripts/ingest_bank_data.py
```

### 5. Running the Agent
Start the agent in development mode:
```bash
uv run python src/agent/main.py dev
```

## 📂 Project Structure

```text
armenian-bank-voice-ai/
├── src/
│   ├── agent/          # Core LiveKit agent and assistant logic
│   ├── audio/          # STT and TTS wrappers
│   ├── rag/            # RAG pipeline: indexing and retrieval
│   ├── scraping/       # Web scrapers for Armenian bank websites
│   ├── vectorstore/    # ChromaDB management logic
│   └── utils/          # Text processing and common utilities
├── config/             # YAML configurations for banks and agent
├── data/               # Raw and processed data storage
├── scripts/            # Data ingestion and utility scripts
├── tests/              # Unit and integration tests
├── pyproject.toml      # Project dependencies (managed by uv)
└── .env                # Environment variables (secret)
```

## 📖 Short Documentation

### RAG Pipeline
The agent uses a two-step RAG process:
1.  **Intent Extraction**: When a user speaks, the agent first uses the LLM to classify the query (Category: Credit/Deposit/Branch, Bank Name, and Scope).
2.  **Contextual Retrieval**: Based on the extracted intent, it queries ChromaDB for relevant Armenian banking documents. These documents are then injected into the LLM's prompt as context.

### Guardrails
The system is strictly scoped. It will politely refuse to answer questions that are not related to banking (e.g., weather, general chat) or if the information is not present in the retrieved context.

### Voice Pipeline
*   **VAD**: Silero VAD is used for precise voice activity detection.
*   **Latency**: The pipeline is optimized for "Round-trip" speed, prioritizing streaming responses to minimize user wait time.

---
*For more detailed implementation details, see [Project Plan](armenian-bank-voice-ai-plan.md).*
