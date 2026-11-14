# KV Cache — Deep Dive

## What Problem Does It Solve?

During generation, a transformer needs to compute **attention** for every new token it produces. Attention means: "look at all previous tokens and decide which ones are relevant."

For each token, the model computes three things:
- **Q (Query)** — "what am I looking for?"
- **K (Key)** — "what do I contain?"
- **V (Value)** — "what information do I carry?"

Attention score = `softmax(Q · Kᵀ) · V`

The problem: when generating token #500, you'd need to recompute **K and V for all 499 previous tokens** from scratch. That's quadratic compute cost.

**The KV Cache fix:** After computing K and V for each token the first time, just **save them**. When token #500 arrives, only compute Q for token #500 and look up the already-cached K, V for tokens 1–499.

```
Without KV cache:  Token 500 → recompute K,V for tokens 1–499 + 500  ← 500x redundant work
With KV cache:     Token 500 → load cached K,V for 1–499, compute only token 500's K,V
```

---

## A Concrete Analogy

Imagine you're writing a book chapter word by word, and after each word you must re-read the entire chapter from scratch to decide the next word.

The KV cache is like keeping a **highlighted summary** of everything you've read. Each time you add a word, you just add its highlight — you never re-read from page 1.

---

## Does KV Cache Depend on Context Capacity or Actual Prompt Length?

**Actual prompt/generation length — not the model's max capacity.**

```
You send:   "What is the capital of France?" (7 tokens)
KV cache:   7 entries — one K,V pair per token, per layer

You send:   A 4,000-token document
KV cache:   4,000 entries — scales exactly with actual tokens
```

The model's `max-model-len` (e.g. 32,768) defines the **ceiling** — the longest conversation it can handle. But the KV cache consumed at any moment equals the **actual tokens in that conversation so far**.

### How vLLM Manages This

vLLM pre-allocates a **KV cache pool** at startup using the remaining VRAM after weights load. It then carves this pool into fixed-size **pages** (each page = 16 tokens by default):

```
Startup:  vLLM calculates how many 16-token pages fit in remaining VRAM
          → e.g. "I have 10.3 GB → 23,500 pages → 376,000 total token slots"

Request arrives with 100-token prompt:
          → allocate 7 pages (112 token slots, 100 used)

As model generates token 101, 102...:
          → new pages allocated on demand from the pool

Request finishes:
          → pages returned to the pool for reuse
```

This is **Paged Attention** — the key vLLM innovation. KV cache behaves like virtual memory: allocated on demand, freed on completion, reused across requests.

> [!IMPORTANT] If the pool is exhausted (all pages in use), new requests wait in queue. This is why `--max-num-seqs` matters — it caps how many concurrent requests can hold pages simultaneously.

---

## Is KV Cache Shared Across Users?

**By default: No. Each request/user owns its own private KV cache pages.**

```
User A: "Tell me a joke"          → 5 pages allocated for User A only
User B: "Explain quantum physics" → 8 pages allocated for User B only
```

Their caches never mix. When User A's request finishes, their pages are freed.

---

## Prefix Caching — When KV Cache IS Shared

vLLM supports **Automatic Prefix Caching (APC)**, enabled with `--enable-prefix-caching`.

The idea: if two requests share an identical prefix (e.g. the same system prompt), the KV cache for that shared prefix is computed once and reused for all requests.

```
System prompt (500 tokens) → KV cache computed once → shared across all users
         ┌──────────────────────────────────────┐
         │  [system prompt KV cache — cached]   │  ← shared block, computed once
         └──────────────────────────────────────┘
User A:  [system KV ↑] + [User A's tokens]       ← only user tokens computed fresh
User B:  [system KV ↑] + [User B's tokens]       ← only user tokens computed fresh
User C:  [system KV ↑] + [User C's tokens]       ← only user tokens computed fresh
```

### Real-world impact

| Scenario | Without Prefix Caching | With Prefix Caching |
|---|---|---|
| 1000-token system prompt, 100 users | 100 × 1000 = 100K tokens of KV cache | 1 × 1000 = 1K tokens shared |
| Time-to-first-token | Full prefill every request | Near-instant for cached prefix |
| VRAM efficiency | Low | High |

> [!TIP] Enable this in production: `--enable-prefix-caching`. It's especially powerful for chatbots with a fixed system prompt, RAG pipelines with a shared document context, or coding assistants with a large codebase prefix.

---

## Multi-turn Conversations

In a chat app, each turn grows the KV cache:

```
Turn 1: User says 50 tokens   → model generates 100 tokens → KV cache: 150 tokens
Turn 2: User says 30 tokens   → model generates 80  tokens → KV cache: 260 tokens
Turn 3: User says 20 tokens   → model generates 60  tokens → KV cache: 340 tokens
```

Each turn, the client sends the **entire conversation history** in the API request. vLLM recomputes the KV cache for all past tokens (unless prefix caching is on, in which case it reuses the cached pages from prior turns).

> [!NOTE] This is why long multi-turn conversations become slower over time — prefill cost grows linearly with conversation length. With prefix caching, only the new tokens need prefill.

---

## Summary

| Question | Answer |
|---|---|
| KV cache size depends on? | **Actual tokens in the request**, not the model's max context capacity |
| Is it shared across users? | **No** by default — each request owns private pages |
| Can it be shared? | **Yes**, with `--enable-prefix-caching` for identical prefixes (e.g. system prompts) |
| What happens when pool fills up? | New requests queue until pages are freed |
| vLLM's key innovation? | **Paged Attention** — KV cache as dynamic virtual memory pages, not pre-allocated per-request |
