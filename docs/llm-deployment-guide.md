# LLM Deployment on Vast.ai A4000 (16GB VRAM)

## How to Calculate if a Model Fits

### The VRAM Budget Formula

$$\text{Total VRAM} = \underbrace{W}_{\text{weights}} + \underbrace{KV}_{\text{kv cache}} + \underbrace{A}_{\text{activations + overhead}}$$

---

### 1. Model Weights (W)

This is the dominant cost and easiest to estimate:

| Precision | Bytes per Parameter | Formula |
|---|---|---|
| FP32 | 4 bytes | `params × 4` |
| FP16 / BF16 | 2 bytes | `params × 2` |
| INT8 | 1 byte | `params × 1` |
| INT4 / GPTQ / AWQ | 0.5 bytes | `params × 0.5` |

**Example — 1B model:**
| Precision | VRAM |
|---|---|
| FP32 | ~4 GB |
| FP16 | ~2 GB |
| INT8 | ~1 GB |
| INT4 | ~0.5 GB |

> [!TIP] vLLM defaults to BF16 on modern GPUs. INT4 (via AWQ/GPTQ) halves VRAM at a small quality cost — great for fitting larger models.

---

### 2. KV Cache (KV)

The KV cache grows with **context length** and **batch size** — this is what limits throughput, not just model size.

$$KV = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes}$$

Where:
- $L$ = number of layers
- $H_{kv}$ = number of KV heads (GQA models have fewer than MHA)
- $d_{head}$ = head dimension
- $S$ = max sequence length
- $B$ = batch size
- $\times 2$ = one K, one V

**Rule of thumb:** KV cache for a 1B model at 4K context, batch=1 ≈ **~0.1–0.3 GB**. It scales linearly with batch size and context length.

> [!IMPORTANT] At long contexts (32K+) or large batches, KV cache can exceed model weights. This is why vLLM uses **paged attention** — it manages KV cache like virtual memory.

---

### 3. Overhead (A)

- CUDA kernels / PyTorch framework: ~**1–1.5 GB**
- Activations during forward pass: small, usually **< 0.5 GB** at inference

---

### Quick Sizing Rule

```
Safe VRAM estimate = (params × bytes_per_param) × 1.2 + 1.5 GB overhead
```

### A4000 (16 GB) Capacity at BF16

| Model Size | VRAM for Weights | Fits? | Headroom for KV cache |
|---|---|---|---|
| 1B | ~2 GB | ✅ Easily | ~13.5 GB — large batches / long ctx |
| 3B | ~6 GB | ✅ | ~9 GB |
| 7B | ~14 GB | ✅ Tight | ~1 GB — small batch only |
| 13B | ~26 GB | ❌ | Need INT4 or multi-GPU |
| 7B INT4 | ~3.5 GB | ✅ Comfortably | ~11 GB |

---

## Recommended ~1B Models to Host

| Model | HuggingFace ID | Params | Context | Strengths |
|---|---|---|---|---|
| **Llama 3.2 1B Instruct** | `meta-llama/Llama-3.2-1B-Instruct` | 1.24B | 128K | Meta's flagship tiny model, instruction-tuned, great baseline |
| **Qwen2.5 1.5B Instruct** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | 32K | Punches above weight, strong on coding & math |
| **SmolLM2 1.7B Instruct** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71B | 8K | HuggingFace's efficient tiny model, fast inference |
| **Gemma 3 1B Instruct** | `google/gemma-3-1b-it` | 1B | 32K | Google's compact model, strong multilingual |
| **TinyLlama 1.1B Chat** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | 2K | Oldest but very fast, good for benchmarking throughput |
| **OLMo 2 1B Instruct** | `allenai/OLMo-2-0425-1B-Instruct` | 1B | 4K | Fully open (data + weights), great for research |
| **Phi-1.5** | `microsoft/phi-1_5` | 1.3B | 2K | Strong reasoning for size, text-only |

> [!NOTE] For the A4000's 16GB you can comfortably go up to **7B models in BF16** or **13B in INT4**. Starting with 1B is great for learning deployment mechanics — inference is instant and you can run high concurrency tests easily.

---

## What to Look at When Evaluating a Model

1. **Parameter count** → determines base VRAM
2. **Precision available** (FP16, BF16, GPTQ, AWQ) → tradeoff between quality and VRAM
3. **Context window** → longer = more KV cache needed
4. **GQA vs MHA** → Grouped Query Attention (Llama 3.2, Qwen2.5) uses fewer KV heads → smaller KV cache = higher throughput
5. **Throughput vs latency** — 1B models can serve 100s of tokens/sec on an A4000; test with vLLM benchmarks

---

## Good Starting Point

```bash
ssh vast-a4000

# Serve Llama 3.2 1B — no gating, downloads instantly
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8080

# Or Qwen2.5 (no HF token needed)
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8080
```

Then from your local machine:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-1.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```
