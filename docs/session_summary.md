# Executive Summary: LLM Serving, Observability, Evaluation & Optimization

**Date**: November 14–15, 2026  
**Hardware Environment**: NVIDIA RTX A4000 (16 GB GDDR6 VRAM, Ampere Architecture, 448 GB/s bandwidth) on Vast.ai  
**Software Stack**: Linux x86_64, CUDA 13.2, PyTorch 2.14.0, vLLM 0.33.0 (V1 Engine), Prometheus v2.54.1, Grafana v11.2.0  

---

## 1. Core Objectives Achieved

Today we covered the full production engineering lifecycle for large language model systems:
1. **Infrastructure & Hosting**: Configured remote headless GPU server, resolved port bindings, and established an OpenAI-compatible vLLM serving stack.
2. **Observability & Real-Time Telemetry**: Deployed Prometheus and Grafana with auto-provisioned dashboards, scraping live vLLM metrics every 2s, and designed a custom telemetry architecture for FastAPI applications.
3. **Multi-Model Architectural Load Testing**: Tested 3 architectures (Qwen2.5-1.5B, SmolLM2-1.7B, TinyLlama-1.1B) under a realistic 90-request heterogeneous workload (short factual, medium code/email, long 700-token RAG).
4. **Comprehensive Quality & Capability Evaluation**: Built and executed a 20-question basic suite (Math, Factual, JSON Format, RAG Grounding) and a 10-test advanced developer suite (Pydantic v2 schemas, prompt injection defense, multi-turn memory, complex IFEval, Indic multilingual evaluation).
5. **Inference Optimization Deep Dive**:
   - **Automatic Prefix Caching (APC)**: Conducted a controlled before-and-after trial on a 1,500-token technical context, demonstrating an **80.9% TTFT latency reduction**.
   - **Speculative Decoding**: Successfully launched a dual-model engine (`Qwen2.5-1.5B` target + `Qwen2.5-0.5B` draft) in a 9.0 GB RAM disk (`/dev/shm`), captured 425 draft cycles via Prometheus, and empirically uncovered **The Speculative Ratio Trap**.

---

## 2. Master Benchmark Scorecard

| Run # | Architecture & Experiment | Concurrency | Aggregate Throughput | Single-User TPOT ($p_{50}$) | TTFT ($p_{50}$) | Composite Quality | Key Discovery |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **#1** | `Qwen2.5-1.5B` (Baseline) | 10 streams | 934.9 tok/s | 9.8 ms | 78.0 ms | — | Baseline multi-stream throughput. |
| **#2** | `Qwen2.5-1.5B` (Saturation) | 35 streams | 2,828.3 tok/s | 10.9 ms | 96.7 ms | — | Continuous batching tripled throughput with only 1.1ms TPOT delta. |
| **#3** | `Qwen2.5-1.5B` (Realistic 90-Req) | 30 streams | 2,022.9 tok/s | 11.3 ms | 48.7 ms | **80.0%** (Adv: 60%) | GQA (2 KV heads) keeps KV memory to 28 KB/tok; passes Pydantic v2 & IFEval. |
| **#4** | `SmolLM2-1.7B` (Realistic 90-Req) | 30 streams | 1,485.8 tok/s | 14.3 ms | 50.5 ms | ~65.0% | MHA attention heads require ~130 KB/tok; memory bandwidth bottlenecked at 30 streams. |
| **#5** | `TinyLlama-1.1B` (Speed Baseline)| 30 streams | **2,959.3 tok/s** | **6.9 ms** | 45.3 ms | 55.0% (Adv: 30%) | Fastest generation (~140 tok/s/user), but fails all math (0/5) and breaks JSON schemas. |
| **#6** | `Qwen2.5-1.5B` (Prefix Caching Trial) | 1 stream (Seq) | — | 9.6 ms | **23.9 ms** *(vs 91.9 ms)* | 100% | **80.9% TTFT drop** on 1,500-token shared prompt via Radix Tree reuse. |
| **#7** | `Qwen2.5-1.5B` (Speculative Decoding) | 1 stream | 6.6 tok/s | 155.9 ms | 77.8 ms | 100% | **Speculative Ratio Trap**: Drafting on 0.5B slowed down a 1.5B model due to low $T_{\text{target}}/T_{\text{draft}}$ ratio. |
| **#8** | `Qwen2.5-1.5B` (RTX 3090 Baseline BF16) | 30 streams | 2,382.7 tok/s | 9.1–9.9 ms | 114.2 ms | **85.0%** | Baseline on 24GB RTX 3090; 2.89 GiB weight VRAM, 15.78 GiB KV cache. |
| **#9** | `Qwen2.5-1.5B-AWQ` (Native AWQ INT4) | 30 streams | 2,484.8 tok/s | 9.3–9.9 ms | 124.1 ms | **75.0%** | Slashes weights by 62% (1.10 GiB); frees +1.79 GiB KV cache. Reference kernels stall at high batch sizes. |
| **#10** | `Qwen2.5-1.5B-AWQ` (Marlin Ampere Kernels) | 30 streams | **2,779.4 tok/s** | **8.1–8.9 ms** | 109.2 ms | **75.0%** | **38% faster single-stream eval (766 tok/s, 8.0ms TPOT)** via async Tensor Core double-buffering. |
| **#11** | `Qwen2.5-0.5B-Custom-AWQ` (Self-Quantized) | 30 streams | **4,004.8 tok/s** | **5.8–6.1 ms** | 49.2 ms | **60.0%** | Quantized from scratch in 76s via AutoAWQ; 438 MB weights, 100% RAG faithfulness. |

---

## 3. Top Engineering Discoveries

### 1. The Power of GQA vs. MHA in High-Concurrency Serving
* SmolLM2-1.7B suffered a ~36% throughput drop compared to Qwen2.5-1.5B under 30 streams.
* **Why**: SmolLM2 uses standard Multi-Head Attention (32 query heads, 32 KV heads), requiring 130 KB per token. Qwen2.5 uses Grouped-Query Attention (12 query heads, 2 KV heads), slashing KV cache to **28 KB per token**. Under heavy batching, memory bandwidth saturation occurs far earlier on MHA models.

### 2. Speed vs. Reliability Tradeoff (The Production Selection Index)
* **TinyLlama-1.1B** broke the throughput record at **2,959 tokens/sec** and sub-7ms TPOT, but scored **0.0% on math** and completely failed structured JSON/Pydantic validation.
* **Qwen2.5-1.5B** delivered **2,022 tokens/sec** while maintaining an **80.0% accuracy score** and 100% pass on Pydantic v2 schemas.
* **Rule**: For agent tool calling, structured JSON output, and reasoning, Qwen is dramatically superior. For keyword extraction, classification, and spell-checking, TinyLlama is an ultra-fast cost-saver.

### 3. Automatic Prefix Caching in Production RAG
* On our Apollo 1,500-token specification, disabling APC forced the GPU to compute 1,500 prompt tokens on every request ($\text{TTFT} = 91.9\text{ ms}$).
* Enabling APC dropped subsequent queries to **23.9 ms** (the remaining time was pure network overhead; GPU prefill was $<2\text{ ms}$).
* **Prompt Structure Rule**: Always keep static context at the front (`[System] -> [Document] -> [User Question]`) so token hashes match in the Radix Tree.

### 4. The Speculative Ratio Trap
* Speculative decoding requires a high target-to-draft latency ratio:
  $$\frac{T_{\text{target}}}{T_{\text{draft}}} \gg K \times (1 - \alpha)$$
* When the target is already small (1.5B takes ~10ms), running a 0.5B draft model takes ~5ms per token. Drafting 5 tokens takes $25\text{ ms}$, plus $10\text{ ms}$ for verification = $35\text{ ms}$ for only ~1.95 accepted tokens ($18\text{ ms/tok}$). Speculative decoding should be reserved for $>14\text{B}$ parameter models.

### 5. Utilizing the 9.0 GB RAM Disk (`/dev/shm`)
* Our root filesystem had only 1.1 GB free space. We discovered and utilized the **9.0 GB `/dev/shm` RAM disk** to download and serve `Qwen2.5-0.5B` at >40 GB/s memory bandwidth without expanding the root container disk.

---

## 4. Next Session Agenda

* **Post-Training Quantization (PTQ)**:
  - Benchmark **AWQ (W4A16)** and **GPTQ (INT4)**.
  - Measure memory bandwidth decode acceleration and VRAM reduction without secondary draft models.
  - Compare accuracy loss vs. throughput gains on Qwen2.5.
