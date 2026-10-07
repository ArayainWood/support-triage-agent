# Write-up — Support Ticket Triage Agent

## Architecture decisions (and why)

- **Agent = tool-calling loop ending in a terminal tool.** The model gathers
  evidence with tools, then must call `submit_triage` (typed by `TriageResult`).
  Forcing the final answer through a typed tool guarantees a complete, structured,
  machine-usable decision instead of prose we would have to parse. The output is
  validated with pydantic, so a malformed decision fails loudly rather than
  silently flowing downstream.
- **Provider-agnostic `LLMClient`.** The loop depends on a small interface, not
  on the OpenAI SDK. That gave us a **deterministic offline mock** for free —
  the whole pipeline (and the test-suite) runs with no key and no network — and
  leaves the door open to other providers or self-hosted models (which the role
  explicitly cares about).
- **Three tools, registry-driven.** `lookup_customer_history`,
  `search_knowledge_base`, `check_service_status`. Adding a capability is just a
  new `Tool` subclass; the agent never changes. `check_service_status` exists
  because sample ticket #2 shows the classic trap — the public status page says
  "operational" while a region is actually down — so the agent must verify, not
  trust.
- **Pluggable RAG retrieval.** Default is a zero-dependency IDF-weighted lexical
  retriever (runs anywhere); `RETRIEVER=embedding` switches to OpenAI embeddings
  + cosine similarity. The KB is where real systems evolve most, so it is the
  seam we made easiest to replace (hybrid search / reranking slot in here).
- **Prompt as a deliverable.** The system prompt encodes the urgency rubric,
  the "read the whole thread / weigh sentiment trajectory" rule, language
  mirroring, and grounding guardrails — kept in one file so it is reviewable and
  tunable without code changes.

## What could go wrong — and how it's handled

| Risk | Mitigation |
|------|------------|
| **Hallucinated facts / promises** (e.g. "we refunded you") | Prompt forbids claims not grounded in tool output; billing KB says never promise refunds for *authorized-but-not-captured* charges; `needs_human_review` + lower confidence when unsure. |
| **Malformed model output** | Terminal tool has a strict schema; result is pydantic-validated; invalid JSON/data raises a clear error. |
| **Retrieval miss / empty KB hit** | Lexical scores are normalised and thresholded; the model is told to escalate rather than guess when evidence is thin. |
| **Tool failure** | `ToolRegistry.dispatch` returns an `{"error": ...}` payload to the model instead of crashing the run. |
| **Infinite / runaway loops** | Hard `MAX_AGENT_STEPS` cap. |
| **Wrong language reply** | Prompt enforces language mirroring; regression-tested on the Thai ticket. |
| **Secret leakage** | Key is read from env by the SDK only; `.env` is git-ignored; nothing is logged. |
| **Under-triaging escalation** | Urgency weighs sentiment *trajectory* and account tier, so a calm-then-furious billing thread and an enterprise outage both land as `critical`. |

## How I'd evaluate this in production

- **Offline eval set.** A labelled corpus of tickets (urgency, action, issue
  type, language) covering billing, outages, how-tos, feature requests, plus
  adversarial/multilingual/ambiguous cases. Track **accuracy/F1 on urgency &
  action**, plus a confusion matrix — the costly error is *under*-escalating a
  critical ticket, so weight recall on `critical` accordingly.
- **Retrieval quality.** Measure recall@k / MRR of `search_knowledge_base`
  against known-relevant docs; alert on empty-hit rate.
- **Grounding / hallucination.** LLM-as-judge (plus spot human review) scoring
  whether each `suggested_response` is supported by tool results; track an
  unsupported-claim rate.
- **Online signals.** Human-override rate (did an agent change the urgency/route?),
  reopen rate, time-to-first-response, and CSAT on auto-responded tickets,
  segmented by language and plan.
- **Safety rails in prod.** Start in *suggest-only* mode (human approves sends),
  shadow-run new prompts/models against the eval set before rollout, log every
  decision with its tool evidence for audit, and alert on confidence drops or
  escalation-rate spikes.
