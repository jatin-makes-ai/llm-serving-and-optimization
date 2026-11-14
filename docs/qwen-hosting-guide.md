# Hosting Qwen2.5-1.5B-Instruct with vLLM

## Step 1 — Start a tmux session (keeps server alive after you disconnect)

```bash
ssh vast-a4000
tmux new -s vllm
```

> To re-attach later: `tmux attach -t vllm`
> Detach without killing: `Ctrl+B` then `D`

---

## Step 2 — Serve the model

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --served-model-name qwen-1.5b
```

---

## Step 3 — Test from your local machine

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-1.5b",
    "messages": [{"role": "user", "content": "What is 2+2? Be concise."}],
    "max_tokens": 50
  }'

# List loaded models
curl http://localhost:8080/v1/models
```

---

## Parameter Walkthrough

| Parameter | Value | Why |
|---|---|---|
| `--host 0.0.0.0` | Listen on all interfaces | Required for the SSH tunnel to reach the server |
| `--port 8080` | Port 8080 | Matches the `LocalForward 8080` in our SSH config — auto-tunneled to your laptop |
| `--dtype bfloat16` | BF16 precision | Native dtype for Qwen2.5; best speed/quality tradeoff on A4000 |
| `--max-model-len` | 32768 | Max context window (tokens). Qwen2.5's max. Reduce to 8192 to free KV cache for more concurrency |
| `--gpu-memory-utilization` | 0.90 | Use 90% of GPU VRAM = 14.4 GB. Rest is safety buffer for CUDA kernels |
| `--max-num-seqs` | 128 | Max concurrent requests being processed. vLLM batches these together |
| `--served-model-name` | qwen-1.5b | Alias used in API calls instead of the full HuggingFace path |

> [!TIP] Omit `--max-model-len` on first run to let vLLM auto-detect. Set it explicitly only if you want to cap context to free VRAM for more concurrent requests.

---

## GPU Memory Breakdown for Qwen2.5-1.5B

### Model Architecture (from config.json)
| Parameter | Value |
|---|---|
| Parameters | ~1.54B |
| Hidden size | 1536 |
| Layers | 28 |
| Attention heads | 12 |
| **KV heads** | **2** (aggressive GQA — key efficiency win) |
| Head dimension | 1536 ÷ 12 = **128** |
| Max context | 32,768 tokens |

---

### 1. Model Weights

$$W = 1.54 \times 10^9 \times 2\ \text{bytes (BF16)} = 3.08\ \text{GB}$$

---

### 2. KV Cache — Per Token

$$KV_{\text{token}} = 2 \times L \times H_{kv} \times d_{head} \times \text{bytes}$$

$$= 2 \times 28 \times 2 \times 128 \times 2\ \text{bytes} = 28{,}672\ \text{bytes} \approx 28\ \text{KB/token}$$

The `×2` out front = one K tensor + one V tensor.

**Why only 2 KV heads?** Qwen2.5 uses aggressive GQA (Grouped Query Attention) — 12 query heads share just 2 KV heads. This cuts KV cache size by **6×** vs standard MHA (which would need 12 KV heads).

| Scenario | Tokens | Batch | KV Cache |
|---|---|---|---|
| Short chat | 4,096 | 1 | 112 MB |
| Long doc | 32,768 | 1 | **896 MB** |
| Concurrent: 4K ctx | 4,096 | 16 | 1.75 GB |
| Concurrent: 4K ctx | 4,096 | 64 | **7 GB** |

---

### 3. Full VRAM Budget on A4000 (16 GB)

```
Total GPU VRAM          = 16.0 GB
Reserved by vLLM (10%)  =  1.6 GB  ← safety margin (--gpu-memory-utilization 0.90)
─────────────────────────────────────
Usable pool             = 14.4 GB
  └─ Model weights      =  3.1 GB
  └─ CUDA / framework   =  1.0 GB (approx)
  └─ KV cache pool      = 10.3 GB  ← vLLM manages this as paged blocks
```

**KV cache pool capacity:**
$$\frac{10.3\ \text{GB}}{28\ \text{KB/token}} \approx 375{,}000\ \text{tokens}$$

This means vLLM can hold ~375K tokens of context in-flight simultaneously across all requests. For example:
- 90 concurrent users × 4K context each = 360K tokens → ✅ fits
- 11 concurrent users × 32K context each = 352K tokens → ✅ fits

> [!NOTE] vLLM uses **Paged Attention** to manage the KV cache pool like virtual memory — pages are allocated/freed dynamically per request, so there's no wasted pre-allocation.

---

## What vLLM Logs on Startup (what to look for)

```
INFO: Loading model weights...
INFO: GPU blocks: 1240, CPU blocks: 0       ← each block = 16 tokens × 28KB = 448KB
INFO: Model loaded. KV cache = 10.3 GB
INFO: Uvicorn running on http://0.0.0.0:8080
```

`GPU blocks × 16 tokens/block = total KV token capacity`. You can verify the math against the numbers above.
