# LLM Serving Monitoring & Observability Guide

Monitoring an LLM API differs fundamentally from traditional REST APIs because generation is **streaming, stateful, and non-deterministic in duration**.

---

## 1. The Core Metrics You Need to Know

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **TTFT** (Time to First Token) | Time from client sending prompt until token 1 arrives | User-perceived latency (spinner duration). Driven by **Prompt length & Prefill compute**. |
| **TPOT** (Time Per Output Token) | Average time taken for each generated token ($(\text{Total} - \text{TTFT}) / \text{Output Tokens}$) | Streaming smoothness / reading speed. Target: ~20–50 ms/token (20–50 tok/s). |
| **ITL** (Inter-Token Latency) | Time delta between consecutive tokens | Jitter / stutter during streaming. High P99 ITL = "glitchy" generation. |
| **E2E Latency** | Full request roundtrip ($T_{\text{finish}} - T_{\text{start}}$) | Batch jobs, non-streaming users, overall system throughput. |
| **Token Throughput** | Output tokens generated per second across all users | Hardware utilization & ROI (how hard is the GPU working). |
| **Queue Time / Requests Waiting** | Time a request spends waiting for KV cache / engine slot | Tells you when you need autoscaling or concurrency throttling. |
| **KV Cache Usage %** | Percentage of allocated PagedAttention blocks in use | Leading indicator for throttling/preemption. Near 100% = danger zone. |

### Why p50, p90, and p99?
- **p50 (Median)**: The typical experience for 50% of your requests.
- **p90 / p95**: Performance when the system is under mild congestion.
- **p99 (Tail Latency)**: The worst 1% of experiences (long prompts, heavy concurrent prefill, or queued requests). In LLM serving, tail latency spikes when prefill and decode compete on the GPU!

---

## 2. The Great News: vLLM Exports Everything Natively!

vLLM has a built-in Prometheus scraper endpoint at **`http://localhost:8000/metrics`**. 

Every single metric you asked about is already instrumented inside vLLM as Prometheus histograms and counters:

```
# TTFT histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} ...

# TPOT histogram
vllm:request_time_per_output_token_seconds_bucket{le="0.02"} ...

# E2E request latency
vllm:e2e_request_latency_seconds_bucket ...

# Concurrency & Queue
vllm:num_requests_running
vllm:num_requests_waiting

# KV Cache health
vllm:kv_cache_usage_perc
vllm:prompt_tokens_cached_total
```

---

## 3. How Prometheus & Grafana Work Together

```
┌─────────────────────────────────┐
│     vLLM (/metrics :8000)       │
│  (Emits counters & histograms)  │
└────────────────┬────────────────┘
                 │ scrapes every 5s
                 ▼
┌─────────────────────────────────┐
│       Prometheus Server         │
│    (Time-Series Database)       │
└────────────────┬────────────────┘
                 │ PromQL queries
                 ▼
┌─────────────────────────────────┐
│       Grafana Dashboard         │
│   (Real-time visual graphs)     │
└─────────────────────────────────┘
```

### How Percentiles (p50, p99) Are Calculated
Prometheus uses **Histogram Buckets** (e.g. counts how many requests finished in `< 10ms`, `< 50ms`, `< 100ms`). 
Grafana queries Prometheus using the `histogram_quantile` function:

- **p99 TTFT**:
  ```promql
  histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))
  ```
- **p50 TPOT**:
  ```promql
  histogram_quantile(0.50, sum(rate(vllm:request_time_per_output_token_seconds_bucket[5m])) by (le))
  ```
- **Output Token Throughput (tok/sec)**:
  ```promql
  sum(rate(vllm:generation_tokens_total[1m]))
  ```

---

## 4. Deep Dive: Our Live Setup on the Remote Server

Here is the exact infrastructure actively running on your rented A4000 GPU instance:

```
                                  REMOTE SERVER (vast-a4000)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  [vLLM Engine] (:8000)                                                                 │
│      │                                                                                 │
│      ├── Exposes OpenAI API (/v1/chat/completions)                                     │
│      └── Exposes Prometheus Metrics (/metrics)                                         │
│               ▲                                                                        │
│               │ HTTP GET every 2 seconds (Pull Scraping)                               │
│               │                                                                        │
│  [Prometheus Server] (:9090)                                                           │
│      ├── Binary: /workspace/monitoring/prometheus/prometheus                           │
│      ├── Config: /workspace/monitoring/prometheus/prometheus.yml                       │
│      ├── Scrapes :8000/metrics and stores data in local TSDB (/prometheus/data)       │
│      └── Evaluates PromQL time-series queries                                          │
│               ▲                                                                        │
│               │ HTTP PromQL Queries                                                    │
│               │                                                                        │
│  [Grafana Visualization] (:3000)                                                       │
│      ├── Binary: /workspace/monitoring/grafana/bin/grafana-server                      │
│      ├── Provisioned Datasource: http://localhost:9090                                 │
│      └── Dashboard JSON: /workspace/monitoring/dashboards/vllm_performance.json        │
│                                                                                        │
└──────────────────────────────────────┬─────────────────────────────────────────────────┘
                                       │
                      SSH Tunneling (-L 3000:... -L 9090:... -L 8000:...)
                                       │
                                       ▼
                             LOCAL BROWSER & CLIENT
              ┌──────────────────────────────────────────────┐
              │  Grafana UI:    http://localhost:3000        │
              │  Prometheus UI: http://localhost:9090        │
              │  vLLM API:      http://localhost:8000        │
              └──────────────────────────────────────────────┘
```

### 1. Prometheus Configuration (`prometheus.yml`)
Located at `/workspace/monitoring/prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 2s      # Scrapes targets every 2 seconds for high resolution
  evaluation_interval: 2s

scrape_configs:
  - job_name: "vllm"
    static_configs:
      - targets: ["localhost:8000"]
```

### 2. Grafana Provisioning
Instead of requiring manual UI configuration in Grafana, we configured **Infrastructure-as-Code** provisioning:
1. `/workspace/monitoring/grafana/conf/provisioning/datasources/prometheus.yaml`: Automatically registers Prometheus (`http://localhost:9090`) as the default datasource on boot.
2. `/workspace/monitoring/grafana/conf/provisioning/dashboards/dashboards.yaml`: Automatically scans `/workspace/monitoring/dashboards/` and loads `vllm_performance.json`.

---

## 5. Prometheus Core Concepts Explained

Prometheus is an open-source **metrics collection and time-series database (TSDB)** originally created at SoundCloud.

### A. Pull vs. Push Architecture
* **Traditional systems (Push)**: The application sends UDP/HTTP packets to a central server (e.g. Datadog agent, StatsD). If your app gets 100,000 requests, it sends 100,000 telemetry messages, potentially overwhelming the network.
* **Prometheus (Pull)**: The application simply maintains running counters and histograms in local memory and hosts a lightweight HTTP route (`/metrics`). Prometheus periodically reaches out (**scrapes**) that route (e.g., every 2s or 15s). 
  * *Advantage*: The application never slows down under load; scraping overhead is completely decoupled from request volume.

### B. The 4 Fundamental Metric Types
Every metric in Prometheus belongs to one of four primitive types:

| Metric Type | Definition | Real-World Example |
|---|---|---|
| **1. Counter** | A cumulative number that **only goes up** (or resets to 0 on restart). | `http_requests_total`, `vllm:generation_tokens_total` |
| **2. Gauge** | A value that can **go up or down** arbitrarily. Represents a momentary snapshot. | `vllm:num_requests_running`, `vllm:gpu_cache_usage_factor`, `system_memory_used_bytes` |
| **3. Histogram** | Samples observations (usually durations or sizes) into configurable buckets. Also records total count and sum. | `vllm:time_to_first_token_seconds_bucket`, `http_request_duration_seconds_bucket` |
| **4. Summary** | Similar to Histogram, but computes streaming client-side $\phi$-quantiles directly. (Histograms are almost universally preferred because Histograms can be aggregated across multiple servers, whereas Summaries cannot). | Client-side SLA latency summaries |

### C. The Time-Series Data Model
In Prometheus, every time-series consists of:
1. **Metric Name**: `vllm:time_to_first_token_seconds_bucket`
2. **Labels (Dimensions)**: `{model_name="qwen-1.5b", le="0.1"}`
3. **Timestamp**: Millisecond epoch
4. **Value**: `float64`

### D. PromQL Fundamentals
* `rate(metric[window])`: Computes average per-second increase of a counter over a sliding window (e.g., `[30s]`). Great for smooth trendlines.
* `irate(metric[window])`: Computes the "instantaneous rate" looking only at the last two data points. Captures sharp spikes and short-burst benchmarks without smoothing dilution.
* `histogram_quantile(0.99, sum(rate(...[1m])) by (le))`: Estimates the 99th percentile across bucket boundaries (`le` = less than or equal).

---

## 6. Grafana Core Concepts Explained

Grafana is an **open-source visualization and dashboard engine**. Grafana does **not** store any metric data itself; it connects to data sources (Prometheus, Postgres, Elasticsearch, InfluxDB) and queries them dynamically.

### Core Concepts:
1. **Data Source**: The backend connection definition. In our case, Grafana connects to Prometheus at `http://localhost:9090`.
2. **Dashboard**: A curated collection of panels arranged in a grid.
3. **Panel**: An individual visual widget on a dashboard. Panels can be:
   * **Time Series (Line Chart)**: Throughput over time, TTFT over time.
   * **Stat / Single Value**: Current Running Requests, Peak Token/s.
   * **Gauge**: KV Cache Usage % (0% to 100% dial).
   * **Heatmap**: Request latency distributions.
4. **Variables / Templating**: Dropdown menus at the top of a dashboard that let you filter by model, instance, or environment dynamically.

---

## 7. How to Instrument a Custom FastAPI Application

If you build your own custom Python/FastAPI microservice (e.g. for pre-processing, custom inference, or a business API), here is the step-by-step production standard for telemetry.

### Method A: The Plug-and-Play Way (`prometheus-fastapi-instrumentator`)

This is the standard 3-line library that auto-instruments all routes, response codes, and request durations:

```bash
pip install fastapi uvicorn prometheus-fastapi-instrumentator
```

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="My Custom ML Service")

# 1. Initialize and expose /metrics
Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    # Your model inference logic here
    return {"prediction": "class_a"}
```

When you start this app with `uvicorn main:app --port 8000`, a `/metrics` route is automatically created that exposes:
* `http_requests_total{handler="/predict", method="POST", status="200"}`
* `http_request_duration_seconds_bucket{le="0.1"}`

---

### Method B: The Full-Control Way (`prometheus_client`)

When building an AI or data service, you usually have domain-specific metrics (e.g. tokens generated, model load time, queue length). Here is how you write custom metrics from scratch:

```python
import time
from fastapi import FastAPI, Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)

app = FastAPI(title="Custom LLM Gateway")

# -------------------------------------------------------------
# 1. Define Your Custom Prometheus Metrics
# -------------------------------------------------------------
# A Counter for total requests processed
TOTAL_INFERENCE_REQUESTS = Counter(
    "custom_inference_requests_total",
    "Total inference requests handled",
    ["model", "status"]
)

# A Counter for tokens generated
TOTAL_TOKENS_GENERATED = Counter(
    "custom_tokens_generated_total",
    "Total tokens streamed to clients",
    ["model"]
)

# A Gauge for requests currently being processed
CONCURRENT_ACTIVE_REQUESTS = Gauge(
    "custom_active_requests",
    "Number of requests currently in-flight"
)

# A Histogram for model latency (with custom bucket boundaries in seconds)
INFERENCE_DURATION = Histogram(
    "custom_inference_duration_seconds",
    "Latency of inference execution",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# -------------------------------------------------------------
# 2. Expose the /metrics Endpoint for Prometheus to Scrape
# -------------------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -------------------------------------------------------------
# 3. Instrument Your Business Logic
# -------------------------------------------------------------
@app.post("/v1/generate")
async def generate(prompt: str, model: str = "qwen-1.5b"):
    # Increment the active concurrency gauge
    CONCURRENT_ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Simulate or execute model inference
        # In reality, this would be your PyTorch/Triton/vLLM call
        time.sleep(0.12)  # simulated inference
        output_tokens = 45

        # Record metrics on success
        TOTAL_INFERENCE_REQUESTS.labels(model=model, status="success").inc()
        TOTAL_TOKENS_GENERATED.labels(model=model).inc(output_tokens)
        
        return {"output": "Simulated generated text", "tokens": output_tokens}
        
    except Exception as e:
        TOTAL_INFERENCE_REQUESTS.labels(model=model, status="error").inc()
        raise e
        
    finally:
        # Record execution duration in the histogram
        duration = time.time() - start_time
        INFERENCE_DURATION.labels(model=model).observe(duration)
        
        # Decrement active concurrency gauge
        CONCURRENT_ACTIVE_REQUESTS.dec()
```

---

### Step 4: Tell Prometheus to Scrape Your FastAPI App

In your `prometheus.yml`, simply add your FastAPI service target:

```yaml
scrape_configs:
  - job_name: "custom_fastapi_service"
    scrape_interval: 2s
    static_configs:
      - targets: ["localhost:8000"]
```

### Step 5: Querying Your FastAPI App in Grafana

Once Prometheus scrapes your FastAPI `/metrics` route, open Grafana and add panels with these queries:

1. **Request Rate (req/sec)**:
   ```promql
   sum(rate(custom_inference_requests_total[1m])) by (status)
   ```
2. **Current Active Users / In-Flight Requests (Gauge)**:
   ```promql
   custom_active_requests
   ```
3. **P95 Latency**:
   ```promql
   histogram_quantile(0.95, sum(rate(custom_inference_duration_seconds_bucket[1m])) by (le))
   ```
4. **Token Generation Throughput (tok/sec)**:
   ```promql
   sum(rate(custom_tokens_generated_total[1m]))
   ```
- Grafana (pre-imported official vLLM dashboard JSON with all charts ready)

### Option 3: Benchmark Test Report (Locust / vLLM Benchmark)
If you want to simulate 10–50 users hammering the server and get an immediate Markdown / CSV summary of p50, p90, p99 TTFT & TPOT without running persistent monitoring infrastructure.
