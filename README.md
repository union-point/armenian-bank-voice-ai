## Armenian Bank Voice AI Agent


Voice AI customer support agent for Armenian banks that answers user queries in Armenian using a strictly scoped RAG pipeline over bank data (Credits, Deposits, Branch Locations).

---

## 🎯 Objective

Build an end-to-end voice assistant using self-hosted LiveKit that:

* understands and speaks Armenian
* answers only from verified bank data
* scales easily to support multiple banks via a config-driven pipeline

---

## 🧱 Tech Stack

* **LiveKit** — low-latency real-time voice streaming (self-hosted, no cloud dependency)
* **Whisper large v3 (via Groq)** — high-quality Armenian STT with fast inference
* **Silero VAD** — efficient speech segmentation for streaming input
* **Gemini 2.5 Flash** — fast and cost-efficient LLM for RAG responses
* **Gemini 2.5 Flash TTS** — natural Armenian speech synthesis
* **LlamaIndex** — document ingestion, chunking, and retrieval pipeline
* **ChromaDB** — lightweight local vector store with metadata filtering
* **paraphrase-multilingual-MiniLM-L12-v2** — multilingual embeddings with good Armenian support

---
## 📂 Project Structure


```text
armenian-bank-voice-ai/
├── src/
│   ├── agent/          # Core LiveKit agent and assistant logic
│   ├── audio/          # STT and TTS wrappers
│   ├── rag/            # RAG pipeline: indexing and retrieval
│   ├── scraping/       # Web scrapers for Armenian bank websites
│   |     ├── utils/          # Text processing and common utilities
|   |     └── scraper.py
│   └── vectorstore/    # ChromaDB management logic
├── config/             # YAML configurations for banks and agent
├── data/               # Raw and processed data storage
├── scripts/            # Data ingestion and utility scripts
├── pyproject.toml      # Project dependencies (managed by uv)
└── .env                # Environment variables (secret)
```
## ⚙️ Setup Instructions
### Prerequisites
Ensure you have `uv` installed on your system.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.  Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/union-point/armenian-bank-voice-ai.git
cd armenian-bank-voice-ai
uv sync
uv run -m src.agent.main download-files
```

### 2. Run infrastructure

```bash
sudo docker run -d --name livekit  --network host  livekit/livekit-server  --dev
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

### 4. Scrape bank data

```bash
uv run -m src.scraping.scraper
```

### 5. Ingest data into RAG

```bash
uv run -m scripts.ingest_bank_data
```

### 6. Run Voice Agent

```bash
uv run -m src.agent.main console
```

---

## 🏗️  Architecture & Decisions

```
CLIENT (Web/Mobile/Telephony)
    WebRTC Audio Stream ←→ LiveKit Server (self-hosted)
            │
            ▼
LIVEKIT AGENT SERVER (Self-hosted )
    ┌──────────────┐   ┌───────────────────────┐    ┌──────────────────┐    ┌────────────┐
    │   Silero     │ → │   STT (whisper-large  │ →  │  gemini-2.5-flash│ →  │ google     │
    │     VAD      │   │   -v3)                │    │     API          │    │  TTS API   │
    │ (Voice Act.  │   │                       │    │  + RAG tool      │    │  Armenian  │
    │  Detection)  │   │                       │    │                  │    │            │
    └──────────────┘   └───────────────────────┘    └────────┬─────────┘    └──────▲─────┘
                                                             │                     │
                                                             ▼                     │
                                                             ┌─────────────────────┘
                                                             │  Function Tool: RAG
                                                             ▼
KNOWLEDGE LAYER
    ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
    │  Scraping       │ →  │  ChromaDB        │ →  │  LlamaIndex       │
    │  Pipeline       │    │  Store           │    │  Semantic         │
    │                 │    │                  │    │  Retrieval        │
    │                 │    │                  │    │                   │
    └─────────────────┘    └──────────────────┘    └───────────────────┘
```

### 1. Audio Pipeline Flow
1. **Input**: Client audio via WebRTC/SIP → Silero VAD detects speech
2. **STT**: whisper-large-v3 processes audio → Armenian text
3. **LLM**: gemini-2.5-flash receives text + system prompt + RAG context → generates response
4. **Guardrails**: System prompt enforces scope (Credits/Deposits/Branches only)
5. **TTS**: gemini tts API generates Armenian speech from response text → streaming audio
6. **Output**: Audio streamed back to client via LiveKit

--- 

### 2. Scraping Strategy

 Config-driven scraper for Armenian banking sites.

 - Supports multiple banks via banks.yaml (no code changes)

 - PDF-first extraction for Credits/Deposits (with HTML fallback)

 - Standard HTML scraping for Branches

 - Outputs clean JSON with metadata (category, bank_name, source_url)

---

### 3. Guardrails — 3 Layers

1. **System Prompt** (primary):
   - Explicit: "Only answer questions about Credits, Deposits, and Branch Locations."
   - Explicit: "If asked about anything else, politely decline in Armenian."
   - Examples of in-scope and out-of-scope questions

2. **Query Classification** (pre-retrieval):
   - Lightweight classifier before RAG to detect scope
   - Out-of-scope → short refusal (no LLM call needed)

3. **RAG Metadata Filtering**:
   - ChromaDB queries include bank_name + category filters
   - Reduces hallucination by only retrieving relevant chunks

