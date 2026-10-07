# Write-up — Support Ticket Triage Agent

## Architecture (and why)

- **Tool-calling loop ending in a terminal tool.** The model gathers evidence with
  tools, then must call `submit_triage` (typed by `TriageResult`). Routing the final
  answer through a typed tool guarantees a complete, structured, validated decision
  instead of prose to parse. If the step cap is hit, the agent *forces* a final
  submit rather than failing the ticket.
- **Provider-agnostic `LLMClient`.** The loop depends on a tiny interface, not the
  OpenAI SDK — which gave us a deterministic offline **mock** (the whole pipeline and
  the tests run with no key/network) and room for other or self-hosted models.
- **Registry-driven tools (3).** `lookup_customer_history`, `search_knowledge_base`,
  `check_service_status`. Adding one is just a new `Tool` subclass; the agent never
  changes. The status check exists because ticket #2 shows the trap: the public page
  says "operational" while a region is actually down.
- **Pluggable RAG retrieval.** Default is a zero-dependency lexical retriever;
  `RETRIEVER=embedding` switches to OpenAI embeddings. The KB is the seam that evolves
  most (hybrid search / reranking slot in here).
- **Prompt as a deliverable** (`prompts.py`): urgency rubric, read-the-whole-thread /
  sentiment-trajectory rule, language mirroring, and grounding guardrails.

## What could go wrong — and the handling

| Risk | Mitigation |
|------|------------|
| Hallucinated facts / promises | Prompt forbids ungrounded claims and separates pending authorizations from captured charges; `needs_human_review` + low confidence when unsure |
| Malformed model output | Typed terminal tool + pydantic validation; scores clamped before validation |
| Model never submits | Step cap → forced final `submit_triage` |
| Retrieval miss | Normalised, thresholded scores; model told to escalate, not guess, on thin evidence |
| Tool failure | Dispatch returns an error payload to the model; the run never crashes |
| Wrong-language reply | Prompt enforces mirroring; regression-tested on the Thai ticket |
| Secret leakage | Key read from env by the SDK only; `.env` git-ignored; nothing logged |

## Evaluating in production

- **Offline eval set** labelled for urgency / action / issue / language (incl.
  adversarial and multilingual cases). Track urgency & action F1 and a confusion
  matrix, weighting recall on `critical` — under-escalation is the costly error.
- **Retrieval quality:** recall@k / MRR for `search_knowledge_base`; alert on
  empty-hit rate.
- **Grounding:** LLM-as-judge plus spot human review scoring whether each reply is
  supported by tool results; track an unsupported-claim rate.
- **Online signals:** human-override rate, reopen rate, time-to-first-response and
  CSAT, segmented by language and plan.
- **Rollout safety:** start suggest-only (human approves sends), shadow-test new
  prompts/models on the eval set, and log every decision with its evidence.
