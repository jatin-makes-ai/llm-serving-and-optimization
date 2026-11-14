# Master Guide: LLM Inference Optimization & Acceleration

Inference optimization is the engineering discipline of maximizing token throughput, minimizing per-token latency (TPOT), reducing Time to First Token (TTFT), and packing the maximum concurrent traffic onto available GPU hardware without compromising model output quality.

---

## 1. The Fundamental Physics: Roofline Model & The Two Phases of LLM Inference

To optimize LLM inference, you must first understand the fundamental physical hardware bottleneck: **The Roofline Model**.

$$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Transferred from VRAM (Bytes)}}$$

```
Compute Throughput
      ▲
      │                    ┌────────────────────────────────── (Compute-Bound Ceiling: Tensor Cores)
      │                   / 
      │                  /  
      │                 /   Prefill Phase (Prompt processing)
      │                /    High batch size, Matrix × Matrix (GEMM)
      │               /
      │              /      Decode Phase (Token generation)
      │             /       Batch size 1–64, Matrix × Vector (GEMV)
      │            /        (Memory-Bandwidth Bound)
      │           /
      └──────────┴─────────────────────────────────────────► Arithmetic Intensity (FLOPs/Byte)
```

### The Two Phases of Every Request

| Phase | Operation | Computational Nature | Hardware Bottleneck | Target Metric to Optimize |
|---|---|---|---|---|
| **1. Prefill (Prompt Processing)** | Processes all $N$ input tokens simultaneously in parallel. | **Compute-Bound** ($\text{FLOPs/Byte} \gg 100$). Uses GEMM (General Matrix Multiply). | GPU Tensor Core compute capacity (TFLOPS). | **TTFT** (Time to First Token) |
| **2. Decode (Token-by-Token Generation)** | Autoregressively generates token $t+1$ given tokens $1 \dots t$. Repeated once per generated token. | **Memory-Bandwidth Bound** ($\text{FLOPs/Byte} \approx 1\text{ to }2$). Uses GEMV (General Matrix-Vector). | GPU Memory Bandwidth (GB/s across the VRAM bus). | **TPOT** (Time per Output Token), **ITL** |

> [!IMPORTANT]
> During the decode phase, to generate **a single token**, the GPU must read **every single parameter of the model** and the entire active **KV cache** from VRAM into the on-chip registers/SRAM. On an NVIDIA RTX A4000 (448 GB/s memory bandwidth), reading a 3 GB model weight takes $\approx \frac{3 \text{ GB}}{448 \text{ GB/s}} = 6.7 \text{ ms}$. Hence, a single user can never exceed $\approx 150 \text{ tokens/sec}$ regardless of compute power unless weights are quantized or batched!

---

## 2. Scheduling & Batching Innovations

### A. Static vs. Dynamic vs. Continuous Batching (Iteration-Level Scheduling)

```
Static Batching (Wasteful):
Stream 1: [==== Token Gen ====] [------------- WAITING FOR STREAM 2 -------------]
Stream 2: [======================== Token Gen ========================] [ DONE ]

Continuous Batching (Orca / vLLM):
Step 1:   [Stream 1, Token 1] [Stream 2, Token 1]
Step 2:   [Stream 1, Token 2] [Stream 2, Token 2]
Step 3:   [Stream 1: DONE   ] [Stream 2, Token 3] -> [Stream 3: NEW ADMISSION, Token 1]
```

* **Static / Naive Batching**: Waits for $B$ requests to arrive, runs them together, and waits until the *slowest/longest* request finishes before accepting new requests. GPU utilization suffers from massive tail padding.
* **Continuous Batching (Iteration-Level)**:
  - Instead of batching at the *request level*, the engine schedules at the **iteration (token) level**.
  - As soon as Request A finishes generation, its slot and KV memory are freed, and Request C is immediately admitted on the very next token step without waiting for Request B.
  - **Empirical Gain**: **$3\times$ to $5\times$ aggregate throughput increase** over naive batching.

---

### B. Chunked Prefill (Sarathi-Serve)

When a 2,000-token prompt arrives while 30 users are actively decoding tokens:
* **Without Chunked Prefill**: The engine pauses all 30 decode streams for 150ms to run the massive 2,000-token prefill. Existing users experience a sudden stutter (huge Inter-Token Latency / ITL spike).
* **With Chunked Prefill (`--enable-chunked-prefill`)**:
  - The 2,000-token prompt is broken into smaller chunks (e.g. 512 tokens).
  - In each iteration, the engine co-schedules a 512-token prefill chunk **alongside** the decode tokens of existing users.
  - **Empirical Gain**: Eliminates the "prefill bubble", smoothing out $p_{99}$ ITL latency spikes by up to **$80\%$**.

---

## 3. KV Cache & Memory Architecture

### A. PagedAttention (The Foundation of vLLM)
In standard PyTorch, memory for the Key-Value (KV) cache of a sequence is allocated contiguously for `max_model_len` (e.g., reserving space for 4,096 tokens upfront).
* **The Problem**: 60% to 80% of VRAM is wasted due to **internal fragmentation** (pre-allocated memory never used) and **external fragmentation** (disjoint free memory holes).
* **The Solution (PagedAttention)**: Inspired by OS Virtual Memory paging.
  - The KV cache is divided into non-contiguous blocks of fixed size (typically 16 tokens).
  - A block table maps logical token positions to physical GPU memory pages.
  - Memory is allocated dynamically one page at a time as tokens are generated.
  - **Empirical Gain**: Zero internal memory waste; allows **$2\times$ to $4\times$ higher concurrency** in the same VRAM footprint.

---

### B. Automatic Prefix Caching (APC / Radix Attention)

In multi-turn chat, system prompts, few-shot examples, and RAG architectures, multiple requests share identical text at the beginning of the prompt.

```
Request 1: [System Prompt (1000 tok)] + [User Query A (50 tok)]  -> Prefills 1,050 tokens
Request 2: [System Prompt (1000 tok)] + [User Query B (40 tok)]  -> REUSES CACHED 1000 TOKENS!
                                                                    Only prefills 40 tokens!
```

* **How it works**: vLLM builds a Radix Tree (trie) over physical KV blocks. If a newly arrived prompt shares tokens with an already computed prefix stored in GPU RAM, the engine skips the $O(N)$ prefill math completely and simply links the existing KV memory blocks to the new request.
* **Empirical Gain**:
  - Prefill time (TTFT) on shared contexts drops from **150ms $\rightarrow$ 3ms** ($>95\%$ latency reduction).
  - Frees up massive GPU compute for decode streams.

---

### C. Architectural Attention Reductions: MHA vs. GQA vs. MQA

The structure of the model's attention heads dictates the memory consumption of its KV cache:

$$\text{KV Cache Bytes per Token} = 2 \times 2 \times \text{layers} \times \text{heads}_{\text{kv}} \times \text{dim}_{\text{head}}$$

* **Multi-Head Attention (MHA)**: Each Query head has its own Key and Value head ($H_{kv} = H_q$). (e.g., LLaMA-1, SmolLM2-1.7B). High VRAM consumption per token.
* **Grouped-Query Attention (GQA)**: Multiple Query heads share a single Key and Value head ($H_{kv} \ll H_q$). (e.g., Qwen2.5, LLaMA-3).
* **Multi-Query Attention (MQA)**: All Query heads share a single Key and Value head ($H_{kv} = 1$). (e.g., Falcon).

| Architecture | Example Model | $H_q / H_{kv}$ Ratio | KV Memory per 1K Tokens | Max Concurrent Streams in 10GB |
|---|---|---|---|---|
| MHA | SmolLM2-1.7B | 32 / 32 (1:1) | ~130 KB | ~35 streams |
| GQA | Qwen2.5-1.5B | 12 / 2 (6:1) | **~28 KB** | **~160 streams** |
| MQA | Falcon-7B | 32 / 1 (32:1) | ~8 KB | ~500 streams |

---

### D. KV Cache Quantization (FP8 / INT8 / INT4)

By default, KV cache values are saved in 16-bit precision (`FP16` or `BF16` = 2 bytes per value).
* Using `--kv-cache-dtype fp8` compresses each stored key and value down to 1 byte (`FP8_E4M3` or `FP8_E5M2`).
* **Gain**: Doubles the number of tokens the KV cache can hold in VRAM, allowing **$2\times$ higher concurrency** or **$2\times$ longer context window** with $<0.5\%$ perplexity loss.

---

## 4. Kernel & Hardware-Level Acceleration

### A. FlashAttention-2 and FlashAttention-3
In standard naive attention:
1. $QK^T$ computes an $N \times N$ attention matrix and writes it back to high-bandwidth VRAM.
2. Softmax reads it from VRAM and writes back.
3. Multiplication with $V$ reads it again.
* **FlashAttention (Dao et al.)**: Uses **IO-Aware SRAM Tiling**. Computes the softmax incrementally using an online normalization trick within fast on-chip SRAM without ever materializing the quadratic $N \times N$ matrix in slow VRAM.
* **Empirical Gain**: **$2\times$ to $4\times$ faster prefill** and quadratic memory consumption reduced to $O(N)$ linear memory.

---

### B. CUDA Graphs

During the token generation (decode) phase:
* Each generated token executes 30 to 80 small GPU kernels (RMSNorm, RoPE, GEMV, SwiGLU, Silu, Add) in quick succession.
* Launching each kernel from Python through the CUDA driver takes $\approx 3 \text{ to } 8 \ \mu\text{s}$ of CPU overhead. When a model step only takes $5\text{ ms}$, CPU kernel launch overhead can account for **30%–50% of the entire step time**!
* **CUDA Graphs (`--enforce-eager` vs default)**:
  - On the first pass, CUDA captures the entire sequence of GPU kernel launches, parameter pointers, and memory dependencies into an execution graph.
  - In subsequent decode steps, Python sends a **single single-instruction hardware launch** to replay the graph.
* **Empirical Gain**: **$15\% \text{ to } 35\%$ reduction in decode latency (TPOT)**.

---

## 5. Algorithmic Acceleration: Speculative Decoding

Speculative decoding exploits the fact that **verifying** $K$ tokens in parallel takes roughly the same time as **generating** 1 token sequentially.

```
Step 1 (Drafting):
A small, lightning-fast "Draft Model" (e.g. 0.5B) greedily emits 4 candidate tokens sequentially:
["The", " capital", " of", " France"] (Takes 4 × 2ms = 8ms)

Step 2 (Parallel Verification):
The large "Target Model" (e.g. 70B or 1.5B) evaluates all 4 tokens in a SINGLE forward pass!
Logprobs check:
  - "The"     -> ACCEPTED (Matches target distribution)
  - "capital" -> ACCEPTED
  - "of"      -> ACCEPTED
  - "France"  -> ACCEPTED + Target generates 1 bonus token: " is"

Total: 5 tokens produced in 1 target model step!
```

* **Mathematical Speedup**: If the draft model acceptance rate is $\alpha \approx 70\%-85\%$, speculative decoding delivers a **$1.8\times$ to $2.8\times$ wall-clock speedup** for single-stream generation.
* **Zero Accuracy Loss**: The verification step mathematically guarantees that the sampled output distribution is identical to the target model alone.

---

## 6. Weight Compression: Post-Training Quantization (PTQ)

Because the decode phase is memory-bandwidth bound, the generation speed of an LLM is directly proportional to how many bytes of weights must be read from VRAM per token.

$$\text{Ideal Generation Speedup} \approx \frac{\text{Bytes per Parameter (Original)}}{\text{Bytes per Parameter (Quantized)}}$$

| Quantization Method | Bit Width | Mechanism | Strengths & Weaknesses |
|---|---|---|---|
| **AWQ (Activation-aware)** | 4-bit (W4A16) | Protects the top 1% salient weight channels that have large activation outliers; quantizes remaining 99% to INT4. | **Industry standard for serving**. Fast, accurate, zero perplexity degradation on 7B+ models. |
| **GPTQ** | 4-bit (W4A16) | Uses second-order Taylor expansion (Hessian matrix) to iteratively adjust unquantized weights as others are rounded. | Excellent static compression; slightly slower kernel decompression on newer architectures than AWQ. |
| **FP8 (Native)** | 8-bit (W8A8 / W8A16) | Native hardware FP8 format supported on Ada Lovelace (RTX 4090/A4000 Ada), Hopper (H100), and Blackwell. | **Fastest compute acceleration**. Both weights and activations run on FP8 Tensor Cores without dequantization overhead. |

* **Empirical Gain**: An INT4 model (W4A16) shrinks weight VRAM footprint by **$3.5\times$**, cutting memory transfer time and boosting decode speed by **$2\times$ to $3\times$** while allowing a 14B model to fit comfortably in 10 GB VRAM!

---

## 7. Master Optimization Matrix

| Technique | Primary Benefit | Phase Impacted | Cost / Trade-Off |
|---|---|---|---|
| **Continuous Batching** | $3\times-5\times$ Throughput | Prefill & Decode | Requires dynamic scheduling engine (vLLM/TGI/SGLang). |
| **PagedAttention** | $2\times-4\times$ Concurrency | Memory | Memory management overhead (negligible). |
| **Prefix Caching** | $>90\%$ TTFT drop on shared prompts | Prefill | Small RAM overhead for radix tree index. |
| **Chunked Prefill** | Eliminates decode jitter / stutters | Decode smoothness | Mild decrease in peak isolated prefill speed. |
| **CUDA Graphs** | $15\%-30\%$ faster TPOT | Decode | Slightly longer initial engine warmup time. |
| **Speculative Decoding**| $1.8\times-2.5\times$ faster single-stream generation | Decode | Consumes extra VRAM to keep draft model in memory. |
| **AWQ / GPTQ (W4A16)** | $2\times-3\times$ faster TPOT & $3.5\times$ less VRAM | Decode & VRAM | Minor quantization loss on sub-3B models. |
| **FP8 KV Cache** | $2\times$ larger context / concurrency | Memory | Requires Ada/Hopper architecture or minor numerical scaling. |
