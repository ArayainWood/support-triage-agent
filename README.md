# Support Ticket Triage Agent

An AI agent that triages incoming customer-support tickets. For each ticket it:

1. **Classifies urgency** — `critical` / `high` / `medium` / `low`
2. **Extracts** key information — product area, issue type, customer sentiment (and its *trend* across messages)
3. **Searches a knowledge base** — pluggable RAG retrieval over FAQs/runbooks
4. **Decides the next action** — `auto_respond`, `route_to_specialist`, or `escalate_to_human`, with a drafted reply **in the customer's own language**

It runs in the terminal (no UI required) and uses OpenAI function-calling with three tools.

---

## Quick start

> Requires **Python 3.10+**. Works on Windows, macOS and Linux.

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS/Linux:    source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. run WITHOUT a key — offline deterministic backend (great for a first look)
python main.py --mock

# 3b. run with the real model — set your OpenAI key first:
#     copy .env.example .env   (Windows)   /   cp .env.example .env   (macOS/Linux)
#     then edit .env and set OPENAI_API_KEY=sk-...
python main.py
```

### Where do I put the API key?

Two options (pick one):

- **`.env` file (easiest):** copy `.env.example` to `.env` and set
  `OPENAI_API_KEY=sk-...`. The file is git-ignored and loaded automatically.
- **Environment variable:**
  - Windows (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`
  - macOS/Linux: `export OPENAI_API_KEY="sk-..."`

No key in code, ever. If no key is set and `--mock` is not used, the program
prints a clear message and exits instead of crashing.

---

## Usage

```bash
python main.py                 # triage all sample tickets (real model)
python main.py --mock          # offline deterministic backend, no key needed
python main.py --ticket t2     # triage only ticket t2 (the Thai outage case)
python main.py --json          # machine-readable JSON output
pytest -q                      # run the test-suite (uses the mock, no key)
```

### Configuration (optional)

All via environment variables / `.env` (defaults in parentheses):

| Variable                 | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `OPENAI_MODEL`           | Chat model (`gpt-4o-mini`)                          |
| `OPENAI_EMBEDDING_MODEL` | Embedding model for semantic KB (`text-embedding-3-small`) |
| `RETRIEVER`              | `lexical` (default) or `embedding`                  |
| `MAX_AGENT_STEPS`        | Max tool-calling rounds per ticket (`8`)            |
| `TEMPERATURE`            | Sampling temperature (`0.2`)                        |
| `USE_MOCK`               | `1` forces the offline backend                      |

### Packaging for submission (zip)

If you submit a zip instead of a Git link, export a **clean** archive so no local
virtual environment or cache is bundled (those are machine-specific and would not
work on another computer):

```bash
git archive --format=zip -o submission.zip HEAD
```

This honors `.gitignore`, so `.venv/`, `.env`, and caches are excluded automatically.
To verify, extract into a fresh folder, create a new venv, `pip install -r requirements.txt`,
then `python main.py --mock` and `pytest -q`.

---

## How it works

```
                 ┌──────────────── TriageAgent loop ────────────────┐
  ticket  ──▶    │  system prompt + ticket  ──▶  LLM                 │
                 │       ▲                         │ tool calls      │
                 │       │  tool results           ▼                 │
                 │   ┌───┴───────────────────────────────┐          │
                 │   │  lookup_customer_history            │  tools   │
                 │   │  search_knowledge_base (RAG)        │          │
                 │   │  check_service_status               │          │
                 │   └─────────────────────────────────────┘          │
                 │   when ready  ──▶  submit_triage (typed) ──▶ result │
                 └──────────────────────────────────────────────────┘
```

- The LLM must finish by calling **`submit_triage`**, a terminal tool typed by
  the `TriageResult` schema — so the final answer is always complete and
  structured, never free-form prose.
- The result is validated with **pydantic** before it is returned.

### Project layout

```
support-triage-agent/
├── main.py                     # entry point  ->  python main.py
├── requirements.txt
├── .env.example                # copy to .env and add your key
├── data/
│   ├── sample_tickets.json      # the 3 provided tickets
│   ├── knowledge_base.json      # mock FAQ / runbook corpus
│   └── customers.json           # mock customer records
├── triage_agent/
│   ├── schemas.py               # pydantic contracts + submit_triage schema
│   ├── prompts.py               # the system prompt (a first-class deliverable)
│   ├── agent.py                 # the tool-calling orchestration loop
│   ├── config.py                # env-driven settings
│   ├── cli.py                   # argument parsing + report printing
│   ├── llm/
│   │   ├── base.py              # provider-agnostic LLM interface
│   │   ├── openai_client.py     # real OpenAI backend
│   │   └── mock_client.py       # deterministic offline backend
│   └── tools/
│       ├── registry.py          # Tool base class + registry/dispatch
│       ├── customer_history.py
│       ├── knowledge_base.py    # pluggable lexical / embedding retrieval
│       └── service_status.py
└── tests/
    └── test_triage.py
```

---

## Design highlights

- **Tools and LLM backends are both pluggable.** Add a tool by subclassing
  `Tool`; swap the model by implementing the `LLMClient` protocol. The agent
  loop never changes — this is the main extensibility lever.
- **RAG retrieval is swappable** between a zero-dependency lexical retriever
  (the default, so it runs anywhere) and OpenAI embeddings (`RETRIEVER=embedding`).
- **Offline mock backend** exercises the whole pipeline with no key — used by
  the tests and by `--mock`, so correctness is verifiable without spend.
- **Multilingual:** replies mirror the customer's language (see Thai ticket `t2`).

See [`WRITEUP.md`](WRITEUP.md) for architecture rationale, failure modes, and the
production evaluation plan.
