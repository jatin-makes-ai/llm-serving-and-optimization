# LLM Deployment, Monitoring, Evaluation & Optimization

A complete hands-on engineering repository documenting large language model (LLM) serving on modern NVIDIA GPU hardware using **vLLM**, **Prometheus**, and **Grafana**, alongside empirical performance benchmarks, accuracy evaluations, and inference optimization experiments.

---

## Repository Structure

```
.
├── docs/                                # In-depth technical guides & benchmark logs
│   ├── session_summary.md               # Complete session recap & executive findings
│   ├── glossary.md                      # Comprehensive glossary of LLM terms & techniques
│   ├── benchmark-runs.md                # Master log of all 7 empirical benchmark runs
│   ├── llm-deployment-guide.md          # Hardware, VRAM sizing, and capacity limits
│   ├── qwen-hosting-guide.md            # vLLM OpenAI-compatible hosting manual
│   ├── kv-cache-explainer.md            # PagedAttention, prefill vs decode, prefix caching
│   ├── monitoring-guide.md              # Prometheus metrics, PromQL queries, and Grafana setup
│   ├── model-quality-framework.md       # Evaluation benchmarks (GSM8K, IFEval, Pydantic)
│   └── inference-optimization-guide.md  # Roofline model, batching, speculative decoding, PTQ
│
├── scripts/                             # Reusable Python benchmarking & evaluation suites
│   ├── load_test.py                     # Baseline multi-concurrency load testing
│   ├── load_test_realistic.py           # 90-request heterogeneous load test (Short/Med/Long RAG)
│   ├── eval_benchmark.py                # 20-question basic quality suite (Math, Factual, Format, RAG)
│   ├── eval_suite_advanced.py           # 10-test advanced suite (Pydantic, injection, IFEval, Indic)
│   ├── benchmark_prefix_caching.py      # Automated Before/After prefix caching benchmark
│   ├── benchmark_speculative.py         # Speculative decoding streaming speed & acceptance rate
│   └── inspect_indic.py                 # Multi-lingual tokenizer & generation inspector
│
└── README.md
```

---

## Quick Reference: Benchmark Results Overview

* **Hardware**: NVIDIA RTX A4000 (16 GB GDDR6 VRAM, 448 GB/s bandwidth)
* **Serving Stack**: vLLM 0.33.0, Prometheus v2.54.1, Grafana v11.2.0

| Model | Batch / Stream Throughput | Single-User TPOT | Accuracy (Basic / Adv) | Recommended Production Use-Case |
|---|:---:|:---:|:---:|---|
| **TinyLlama-1.1B** | **2,959 tok/s** | **6.9 ms** | 55.0% / 30.0% | Embedding reranking, keyword classification, high-volume edge tasks. |
| **Qwen2.5-1.5B** | 2,022 tok/s | 10.5 ms | **80.0% / 60.0%** | Structured JSON extraction, Pydantic tool-calling, multi-turn chat, RAG. |
| **SmolLM2-1.7B** | 1,485 tok/s | 14.3 ms | ~65.0% / ~40.0% | General lightweight text generation (MHA head dimension overhead). |

---

## Optimization Breakthroughs
* **Automatic Prefix Caching (APC)**: **80.9% TTFT latency reduction** ($91.9\text{ ms} \rightarrow 23.9\text{ ms}$) on recurring 1,500-token technical context.
* **Speculative Decoding**: Empirically proved the **Speculative Ratio Trap**—why pairing a 0.5B draft with a 1.5B target degrades throughput, and why it is reserved for $>14\text{B}$ parameter targets.
* **In-Memory RAM Disk (`/dev/shm`)**: Served the draft model from the 9.0 GB RAM disk at >40 GB/s bandwidth, completely bypassing container disk limits.
