# Armenian Bank Voice AI Agent - Project Plan

## 1. Project Overview

**Goal**: Build a real-time voice AI customer support agent for Armenian banks using the open-source LiveKit framework. The agent understands and responds in Armenian, answering questions about Credits, Deposits, and Branch Locations only.

**Guiding Decisions** (from requirements):
- ASR: whisper-small-armenian (local, fine-tuned for Armenian)
- LLM: gemini-2.5-flash (fast, multilingual, cost-effective)
- TTS: ElevenLabs API
- VectorDB: ChromaDB
- Embedding Model: armenian-text-embeddings-2-base
- Deployment: Fully self-hosted (LiveKit)
- Data: Hybrid (automated scraping)

---

## 2. System Architecture

```
CLIENT (Web/Mobile/Telephony)
    WebRTC Audio Stream ←→ LiveKit Server (self-hosted)
            │
            ▼
LIVEKIT AGENT SERVER (Self-hosted )
    ┌──────────────┐   ┌───────────────────────┐    ┌──────────────────┐    ┌────────────┐
    │   Silero     │ → │   STT (whisper-small │ →  │  gemini-2.5-flash│ →  │ ElevenLabs │
    │     VAD      │   │   -armenian)         │    │     API          │    │  TTS API   │
    │ (Voice Act.  │   │                      │    │  + RAG tool      │    │  Armenian  │
    │  Detection)  │   │                      │    │                  │    │            │
    └──────────────┘   └───────────────────────┘    └────────┬─────────┘    └──────▲─────┘
                                                       │                   │
                                                       ▼                   │
                                              ┌────────────────────────────┘
                                              │  Function Tool: RAG
                                              ▼
KNOWLEDGE LAYER
    ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
    │  Scraping       │ →  │  ChromaDB       │ →  │  LlamaIndex       │
    │  Pipeline       │    │  Store        │    │  Semantic         │
    │  (Scrapy +      │    │  hosted)         │    │  Retrieval        │
    │  BeautifulSoup) │    │                  │    │                   │
    └─────────────────┘    └──────────────────┘    └───────────────────┘
```

### Audio Pipeline Flow
1. **Input**: Client audio via WebRTC/SIP → Silero VAD detects speech
2. **STT**: whisper-small-armenian processes audio → Armenian text
3. **LLM**: gemini-2.5-flash receives text + system prompt + RAG context → generates response
4. **Guardrails**: System prompt enforces scope (Credits/Deposits/Branches only)
5. **TTS**: ElevenLabs API generates Armenian speech from response text → streaming audio
6. **Output**: Audio streamed back to client via LiveKit

---

## 3. Technology Stack

| Component | Choice | Justification | Latency Target |
|-----------|--------|---------------|----------------|
| **Framework** | LiveKit Agents (Python) | Open-source, self-hostable, proven voice pipeline | - |
| **STT (ASR)** | `whisper-small-armenian` | Fine-tuned for Armenian, runs locally | ~1-2s |
| **VAD** | Silero VAD | Best open-source VAD, multilingual, runs in-process | ~50ms |
| **Turn Detection** | LiveKit Multilingual Turn Detector | Transformer-based, understands when user finishes speaking | ~200ms |
| **LLM** | gemini-2.5-flash | Fast, multilingual, cost-effective, large context | ~1-2s |
| **TTS** | ElevenLabs API | API-based fast, supports multilingual voices | ~300ms |
| **Vector DB** | ChromaDB | Simple, embedded vector database, easy to use | - |
| **RAG Framework** | LlamaIndex | Best-in-class for RAG pipelines, easy LiveKit integration | - |
| **Embeddings** | `armenian-text-embeddings-2-base` | Armenian-specific embedding model | - |
| **Orchestration** | Docker Compose (dev) | Containerized self-hosted deployment | - |

### Fallback Chain
- **TTS**: ElevenLabs API → Google Cloud TTS (verify Armenian support)
- **LLM**: gemini-2.5-flash → gemini-1.5-flash
---

## 4. Module Structure

```
armenian-bank-voice-ai/
├── src/
│   ├── agent/
│   │   ├── main.py              # LiveKit entry point (AgentSession, WorkerOptions)
│   │   ├── assistant.py         # Core Agent class with RAG function tool
│   │   ├── config.py            # Config dataclasses (from YAML/env)
│   │   └── prompts.py           # System prompts (Armenian + scope enforcement)
│   ├── audio/
│   │   ├── stt.py               # whisper-small-armenian STT wrapper
│   │   ├── tts.py               # Elevenlabs TTS wrapper
│   │   └── vad.py               # Silero VAD configuration
│   ├── rag/
│   │   ├── pipeline.py          # LlamaIndex pipeline (chunking, indexing)
│   │   ├── retriever.py         # Query routing + semantic search
│   │   ├── embedder.py          # Embedding model wrapper
│   │   └── reranker.py          # Cross-encoder reranking (optional)
│   ├── scraping/
│   │   ├── base_scraper.py      # Base scraper class with retry/logging
│   │   ├── banks/
│   │   │   ├── ameribank.py
│   │   │   ├── conversebank.py
│   │   │   └── acba.py
│   │   ├── parsers.py           # HTML parsing utilities
│   │   └── scheduler.py         # APScheduler for periodic scraping
│   ├── vectorstore/
│   │   ├── chroma_client.py     # ChromaDB connection + collection management
│   │   └── models.py            # Document/Chunk dataclasses with metadata
│   ├── api/
│   │   ├── routes.py            # FastAPI routes
│   │   └── middleware.py        # Logging, CORS, rate limiting
│   └── utils/
│       └── armenian_text.py     # Armenian text normalization
├── data/
│   ├── raw/                     # Raw scraped HTML/markdown
│   └── processed/              # Chunked documents ready for indexing
├── config/
│   ├── banks.yaml               # Bank URLs, selectors, scraping rules
│   └── agent.yaml               # LLM/TTS/STT model choices
├── tests/
│   ├── unit/                    # ASR, TTS, RAG, guardrail tests
│   ├── integration/             # Full pipeline tests
│   └── eval/                    # LLM judge, metrics tracking
├── infra/
│   ├── docker/
│      ├── Dockerfile.agent
│      ├── Dockerfile.scraper
│      └── docker-compose.yaml
├── scripts/
│   ├── download_models.sh
│   ├── ingest_bank_data.py
│   └── eval_runner.py
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## 5. Data Ingestion & Retrieval Pipeline

### 5.1 Data Model

Each document chunk stored in ChromaDB:
```python
@dataclass
class BankDocument:
    id: str
    bank_name: str           # "Ameriabank", "Converse Bank", etc.
    category: str            # "credit", "deposit", "branch"
    subcategory: str         # "consumer_loan", "mortgage", "term_deposit"
    title: str               # Armenian title
    content: str             # Armenian text content
    url: str                 # Source URL
    scraped_at: datetime
    chunk_index: int
```

### 5.2 Scraping Strategy

**Static pages** (weekly manual review + automated scraping):
- Credit product pages (rates, terms, eligibility)
- Deposit product pages (interest rates, conditions)
- Branch/ATM locator pages
- Tariff pages

**Dynamic pages** (automated):
- Exchange rates (currency, gold)
- Current interest rates

**Scraping approach**:
- Each bank gets a dedicated scraper class in `scraping/banks/`
- Bank scraper inherits from `BaseScraper` (retry, rate limiting, logging)
- Output: Structured JSON → validated → chunked → embedded → indexed
- Adding a 4th bank: Implement interface, add entry to `banks.yaml` — no other changes needed

### 5.3 RAG Retrieval Flow

```
User Query (Armenian text)
       ↓
  Query Routing (LlamaIndex)
       ├→ "credit" → filter to credit chunks
       ├→ "deposit" → filter to deposit chunks
       └→ "branch" → filter to branch chunks
       ↓
  Semantic Search (ChromaDB)
       - Embed query with armenian-text-embeddings-2-base
       - Top-10 retrieval with bank + category filter
       ↓
  Context Assembly
       - Top-3 chunks → formatted prompt
       - Include bank name, category labels
       ↓
  LLM Response Generation
       - System prompt: Armenian instructions + scope guardrails
       - Retrieved context injected
       - Generate Armenian response
```

### 5.4 Update Schedule

| Data Type | Frequency | Method |
|-----------|-----------|--------|
| Credit products | Weekly | Manual review + automated scraping |
| Deposit rates | Daily | Automated scraping |
| Branch locations | Monthly | Automated scraping |
| Exchange rates | Every 4 hours | Automated scraping |
| Tariffs | Weekly | Manual + automated |

---

## 6. Core Components Detail

### 6.1 Agent Session (LiveKit)

```
AgentSession
├── Silero VAD (voice activity detection)
├── Turn Detector (multilingual transformer)
├── STT: whisper-small-armenian
├── LLM: gemini-2.5-flash via API with:
│   ├── System prompt (Armenian, scope guardrails)
│   └── RAG function tool (retrieves bank data)
└── TTS: ElevenLabs API
```

### 6.2 RAG Function Tool

```python
@function_tool
async def query_bank_info(query: Annotated[str, "Bank-related question in Armenian"]) -> str:
    """Retrieve information about Armenian bank products, services, or branches."""
    # 1. Route query to correct category
    # 2. Semantic search in ChromaDB
    # 3. Format top results as context
    # 4. Return formatted context string
```

The LLM decides when to call this tool based on the user's question. Out-of-scope queries are refused by the system prompt.

### 6.3 Guardrails — 3 Layers

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

### 6.4 Armenian Language Handling

**Challenges**: Armenian is low-resource. Mitigation:
- System prompts written in Armenian with explicit examples
- RAG chunks stored with Armenian content
- whisper-small-armenian fine-tuned for Armenian
- Text normalization: Armenian punctuation, AMD currency formatting

---

## 7. Conversation Flow Examples

### Deposit inquiry:
```
Agent: "Ողջույն, հաճախորդ: Ես ձեր վարկային համակարգի օգնականն եմ: Ինչո՞վ կարող եմ օգնել:"
User: "Ինչպիսի՞ ավանդներ ունեք AMD-ով 6 ամսով?"
Agent: (RAG retrieves deposit products)
Agent: "Ameriabank-ը առաջարկում է 6-ամսյա ավանդ AMD-ով 7.5% տոկոսադրույքով..."
```

### Out-of-scope:
```
User: "Որքա՞ն է ձեր աշխատակիցների աշխատավարձը:"
Agent: "Ցավոք, այդ տեղեկատվությունը հասանելի չէ: Ես կարող եմ օգնել միայն վարկերի, ավանդների և մասնաճյուղերի հարցերում:"
```

---

## 8. Latency Optimization

| Stage | Target | Optimization |
|-------|--------|--------------|
| VAD | < 50ms | Silero runs in-process, no network |
| ASR | < 2s | whisper-small-armenian, batch when possible |
| LLM (thinking) | < 2s | gemini-2.5-flash with streaming, parallel RAG retrieval |
| RAG retrieval | < 500ms | ChromaDB indexing, pre-warmed |
| TTS | < 500ms | ElevenLabs, (streaming audio chunks?) |
| **Total E2E** | **< 5s** | Streaming: TTS starts before LLM finishes |

**Streaming TTS**: ElevenLabs generates audio in chunks; first audio plays before full text is generated.

**Parallel execution**: While LLM generates, pre-fetch TTS voice model weights.

---

## 9. Evaluation Metrics & Test Strategy

### Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| WER (Armenian ASR) | < 25% | Test set of 100 Armenian audio samples |
| Refusal accuracy | > 95% | 50 out-of-scope test queries |
| Retrieval precision@3 | > 85% | RAG eval set of 200 Q&A pairs |
| E2E latency (P95) | < 5s | Real conversation measurement |
| Guardrail pass rate | > 98% | Adversarial test set |
| Armenian fluency | > 4/5 | LLM-as-judge scoring |

### Test Types

- **Unit Tests**: STT accuracy, RAG retrieval precision, guardrail refusal, TTS validity
- **Integration Tests**: Full STT→RAG→LLM→TTS pipeline, scraper output, LiveKit connection
- **LLM Judge Eval**: Factuality, naturalness, refusal correctness via Claude/GPT-4o judge

### Test Data

Collect Armenian audio samples (100+) covering:
- Credit questions (consumer, mortgage, car)
- Deposit questions (term deposits, savings)
- Branch/location questions
- Out-of-scope questions
- Noisy conditions (background noise, accents)

---

## 10. Project Phases & Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Set up project structure, pyproject.toml, requirements.txt
- [ ] Configure LiveKit Agent SDK with self-hosted LiveKit server
- [ ] Integrate Silero VAD + turn detection
- [x] Integrate whisper-small-armenian STT (local)
- [ ] Integrate ElevenLabs API TTS
- [ ] Verify audio pipeline: VAD → STT → text, text → TTS → audio
- [ ] Unit tests for audio pipeline

### Phase 2: RAG & Knowledge Layer (Week 2-3)
- [ ] Set up ChromaDB (local/embedded)
- [ ] Write scrapers for 3 banks (Ameriabank, Converse Bank, Unibank)
- [ ] Build LlamaIndex RAG pipeline (chunk, embed, index)
- [ ] Implement RAG function tool
- [ ] Build embedding pipeline (armenian-text-embeddings-2-base)
- [ ] Tests: scraper validation, retrieval precision

### Phase 3: LLM & Guardrails (Week 3-4)
- [ ] Write Armenian system prompts with scope enforcement
- [ ] Implement query classification (in-scope vs out-of-scope)
- [ ] Build refusal responses in Armenian
- [ ] Full conversation flow: STT → RAG → LLM → TTS
- [ ] Tests: guardrail pass rate, refusal accuracy

### Phase 4: Polish & Integration (Week 4-5)
- [ ] Automated data refresh scheduler

---

## 11. Resource Requirements

### Hardware

| Component | Dev (Local) | Prod (Self-hosted) |
|-----------|-------------|-------------------|
| Agent Server | 4-core CPU, 16GB RAM | 8-core CPU, 32GB RAM, GPU (T4+) |
| ChromaDB | 2-core, 8GB RAM | 4-core, 16GB RAM (SSD) |
| LiveKit Server | 4-core, 8GB RAM | 8-core, 16GB RAM |
| Redis (optional) | — | 2-core, 4GB RAM |

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| whisper-small-armenian WER too high | Medium | High | Fine-tune on collected audio; use larger model if needed |
| Bank websites block scraping | Medium | Medium | Respect robots.txt; explore data partnerships |
| LLM hallucinations about bank data | Low | High | Strict RAG grounding; never answer from training data alone |
| LiveKit self-hosted complexity | High | Medium | Docker Compose first;  |
| Armenian language edge cases | Medium | Low | Build robust test set; iterate on prompts |
| Low-resource language limitations | Medium | Medium | Heavy reliance on RAG; fine-tune if needed |

---

## 13. Next Steps

1. **This week**: Set up project skeleton, verify audio pipeline (VAD → STT → text, text → TTS → audio)
2. **Initial bank data scrape**: Pick one bank (Ameriabank), extract credit/deposit/branch data
3. **MVP wire-up**: Connect STT → gemini-2.5-flash → ElevenLabs TTS with basic Armenian greeting prompt
4. **Decision point**: Evaluate whisper-small-armenian WER; adjust stack if needed

---

## 14. Quick-Start Commands

```bash
# Install dependencies
uv init --bare
uv add "livekit-agents[silero,turn-detector]~=1.4" python-dotenv llama-index chromadb transformers

# Download models
python scripts/download_models.sh

# Start infrastructure
docker compose -f infra/docker/docker-compose.yaml up -d

# Run agent locally
uv run python -m src.agent.main dev

# Ingest bank data
uv run python scripts/ingest_bank_data.py

# Run tests
uv run pytest tests/ -v
```
