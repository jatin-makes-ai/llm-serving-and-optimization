# Master Glossary: LLM Serving, Telemetry & Optimization

A comprehensive reference of core terms, mathematical definitions, architectural mechanisms, and concrete examples encountered in production LLM deployment.

---

## 1. Latency & Performance Metrics

### TTFT (Time to First Token)
* **Definition**: The time elapsed between a client sending an HTTP prompt request and receiving the very first generated token chunk.
* **Driven by**: Prompt length, GPU prefill compute (GEMM), network roundtrip, and queue time.
* **Formula**:
  $$\text{TTFT} = T_{\text{token 1 arrived}} - T_{\text{request dispatched}}$$
* **Target**: $<100\text{ ms}$ for interactive chatbots; $<30\text{ ms}$ when prefix caching hits.

### TPOT (Time Per Output Token)
* **Definition**: The average time taken to generate each subsequent token after the first token has arrived. Measures streaming smoothness and reading pace.
* **Formula**:
  $$\text{TPOT} = \frac{T_{\text{stream finished}} - T_{\text{first token}}}{\text{Total Output Tokens} - 1}$$
* **Target**: $10\text{--}25\text{ ms/token}$ ($40\text{--}100\text{ tokens/sec}$ per user). Fast human reading speed is $\approx 5\text{--}8\text{ tokens/sec}$.

### ITL (Inter-Token Latency)
* **Definition**: The time delta between consecutive token chunk emissions ($T_{i} - T_{i-1}$).
* **Why it matters**: While TPOT is an average, ITL measures **jitter and stutter**. If $p_{99}$ ITL spikes due to incoming prefill bursts, the user sees the stream freeze and stutter.

### E2E Request Latency
* **Definition**: Total roundtrip time from dispatch to the final `[DONE]` token.
* **Formula**:
  $$\text{E2E} = \text{TTFT} + (\text{Output Tokens} - 1) \times \text{TPOT}$$

### Token Throughput
* **Definition**: Total tokens generated per second across **all concurrent streams** on the GPU.
* **Formula**:
  $$\text{Throughput} = \frac{\sum \text{Generated Tokens}}{\text{Wall-Clock Time (seconds)}}$$
* **Example**: On an RTX A4000, 1 single stream yields $\approx 95\text{ tok/s}$. Under 30 concurrent streams with continuous batching, aggregate throughput reaches $\approx 2,022\text{ tok/s}$ on Qwen-1.5B and $\approx 2,959\text{ tok/s}$ on TinyLlama.

---

## 2. Hardware Bottlenecks & Execution Physics

### The Roofline Model
* **Definition**: A hardware performance model stating whether an operation is limited by raw compute capability (TFLOPS) or memory bandwidth (GB/s).
* **Arithmetic Intensity**:
  $$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Transferred from VRAM (Bytes)}}$$

### Prefill Phase (Prompt Processing)
* **Definition**: The first stage of inference where the entire prompt is ingested simultaneously.
* **Nature**: **Compute-Bound** (High Arithmetic Intensity). Operates as a dense General Matrix Multiply (GEMM) utilizing Tensor Cores at high utilization.

### Decode Phase (Token Generation)
* **Definition**: The autoregressive stage where tokens are generated one at a time.
* **Nature**: **Memory-Bandwidth Bound** (Low Arithmetic Intensity $\approx 1\text{ FLOP/Byte}$). Every single step requires reading the entire parameter weight tensor and the KV cache across the VRAM memory bus.

---

## 3. Serving Architecture & Scheduling

### Continuous Batching (Iteration-Level Scheduling)
* **Definition**: An asynchronous scheduling mechanism (introduced by Orca / vLLM) that batches requests at the individual token generation step rather than at the request level.
* **Static vs. Continuous**:
  * *Static*: Locks $B$ requests together until the longest finishes; short requests sit idle in VRAM.
  * *Continuous*: As soon as Request A finishes generation, its slot is freed, and incoming Request C is admitted on the very next token iteration.
* **Gain**: $3\times\text{--}5\times$ higher throughput than static batching.

### Chunked Prefill (Sarathi-Serve)
* **Definition**: Splitting large input prompts into smaller token chunks (e.g., 512 tokens) and co-scheduling them with ongoing decode steps.
* **Why it works**: Attention is causal (unidirectional). Tokens 201–400 only attend to tokens 1–400. Computing Chunk 1 first, saving its Key-Values to the cache, and then computing Chunk 2 with access to Chunk 1's cache yields the exact same mathematical output.
* **Gain**: Completely eliminates the "prefill bubble" and prevents decode latency stuttering ($80\%$ reduction in $p_{99}$ ITL).

---

## 4. KV Cache & Memory Management

### KV Cache (Key-Value Cache)
* **Definition**: Cached representations of past tokens' Key and Value attention matrices stored in GPU VRAM so earlier tokens do not need to be recomputed on every autoregressive step.

### PagedAttention
* **Definition**: An algorithm developed for vLLM inspired by virtual memory paging in operating systems.
* **Mechanism**: Divides the continuous KV cache into fixed-size physical memory pages (typically 16 tokens/block). Eliminates contiguous memory allocation, internal fragmentation, and external fragmentation.
* **Gain**: Recovers $60\%\text{--}80\%$ of previously wasted VRAM, doubling or tripling concurrent stream capacity.

### Automatic Prefix Caching (APC / Radix Attention)
* **Definition**: A caching mechanism that matches identical prompt prefixes across requests using a Radix Tree (Trie) over memory blocks.
* **Gain**: On shared system prompts, multi-turn conversations, or repeated RAG contexts, TTFT drops from $\sim 120\text{ ms} \rightarrow \mathbf{\sim 23\text{ ms}}$ (our empirical test achieved an **80.9% latency reduction**).

### Attention Architectural Variants

$$\text{KV Bytes per Token} = 2 \times 2 \times \text{layers} \times \text{heads}_{\text{kv}} \times \text{dim}_{\text{head}}$$

* **MHA (Multi-Head Attention)**: Equal number of Query heads and Key/Value heads ($H_q = H_{kv}$). High memory footprint (e.g. SmolLM2-1.7B).
* **GQA (Grouped-Query Attention)**: Multiple Query heads share a single Key/Value head ($H_{kv} \ll H_q$). Dramatically smaller KV cache (e.g. Qwen2.5-1.5B has 12 query heads sharing 2 KV heads, consuming only 28 KB/token).
* **MQA (Multi-Query Attention)**: All Query heads share a single KV head ($H_{kv} = 1$). Lowest memory footprint (e.g. Falcon).

---

## 5. Hardware Acceleration & Compilers

### FlashAttention (Dao et al.)
* **Definition**: An exact, IO-aware attention algorithm that computes attention incrementally using online softmax normalization within fast on-chip SRAM, avoiding materializing the $N \times N$ matrix in slow high-bandwidth memory (HBM).
* **Gain**: $2\times\text{--}4\times$ faster prefill; memory complexity reduced from $O(N^2)$ to $O(N)$.

### CUDA Graphs
* **Definition**: A CUDA driver feature that records a sequence of kernel launches and memory dependencies during warmup, replaying them with a single CPU instruction during inference.
* **Gain**: Eliminates CPU kernel launch latency ($3\text{--}8\ \mu\text{s}$ per kernel across 50 kernels), reducing generation TPOT by $15\%\text{--}30\%$.

---

## 6. Algorithmic Optimization & Compression

### Speculative Decoding (Draft & Verify)
* **Definition**: Using a tiny, fast "Draft Model" (e.g. 0.5B) to propose $K$ candidate tokens, which a larger "Target Model" (e.g. 70B) verifies in a single parallel forward pass.
* **Acceptance Rate ($\alpha$)**: The fraction of drafted tokens approved by the target model.
* **The Speculative Ratio Trap**:
  * If Target is $10\times$ slower than Draft (e.g., 70B vs 8B) $\rightarrow$ **Huge 2x speedup**.
  * If Target is only $2\times$ slower than Draft (e.g., 1.5B vs 0.5B) $\rightarrow$ **Slowdown!** Drafting 5 tokens takes longer than generating 1 token directly on the target.

### Prompt Lookup Decoding (ngram)
* **Definition**: A draft-model-free speculative decoding technique that searches the input prompt for repeating n-grams to guess candidate future tokens.
* **Cost**: 0 MB extra VRAM, zero secondary neural network overhead. Ideal for code editing, summarization, and RAG.

### Quantization (PTQ - Post-Training Quantization)
* **W4A16 (Weight-Only 4-bit)**: Model weights are stored in 4-bit INT; unpacked to FP16 in registers to multiply with FP16 activations.
* **AWQ (Activation-aware Weight Quantization)**: Protects the top 1% salient weight channels that have outlier activations; quantizes the remaining 99%.
* **GPTQ**: Uses second-order Taylor expansion (inverse Hessian matrix) to compensate for quantization error.
* **FP8 (Floating Point 8)**: Native 8-bit format (`E4M3` / `E5M2`) accelerated by modern Tensor Cores (Ada Lovelace, Hopper).

---

## 7. Observability & Telemetry

### Prometheus
* **Definition**: An open-source, pull-based metrics collection engine and time-series database (TSDB). Stores numerical pairs of `(timestamp, float64)` indexed by metric name and labels.
* **The 4 Metric Types**:
  1. **Counter**: Monotonically increasing number (e.g., `vllm:generation_tokens_total`).
  2. **Gauge**: Momentary snapshot that goes up and down (e.g., `vllm:num_requests_running`).
  3. **Histogram**: Latency observations binned into configurable threshold buckets (`le="0.1"`).
  4. **Summary**: Client-side streaming quantiles.

### PromQL Functions
* **`rate(counter[window])`**: Calculates average per-second change over a time window.
* **`irate(counter[window])`**: Instantaneous rate looking only at the last two data points (captures bursty load spikes without smoothing).
* **`histogram_quantile(φ, sum(rate(..._bucket[window])) by (le))`**: Calculates estimated percentiles ($p_{50}, p_{90}, p_{99}$) from histogram bucket distributions.

### Grafana
* **Definition**: An open-source visualization dashboard engine that executes PromQL queries against Prometheus and renders real-time time-series graphs, stat gauges, and heatmaps.
