import asyncio
import time
import json
import statistics
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen-1.5b"
CONCURRENCY = 35
TOTAL_REQUESTS = 100

PROMPTS = [
    "Explain quantum computing in simple terms for a high school student.",
    "Write a short story about an astronaut discovering a strange plant on Mars.",
    "What are the main differences between Python and Rust? List 3 key points.",
    "Give me a step-by-step recipe to make authentic Italian carbonara.",
    "Explain how the Transformer attention mechanism works without heavy math.",
    "Describe the process of photosynthesis and why it is essential for life.",
    "Draft a professional email asking a client for a project deadline extension.",
    "What causes inflation in an economy? Explain the core drivers briefly."
]

async def send_streaming_request(session, req_id, prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "stream": True,
        "temperature": 0.7
    }
    
    t_start = time.perf_counter()
    ttft = None
    output_tokens = 0
    t_first_token = None
    
    try:
        async with session.post(URL, json=payload) as resp:
            if resp.status != 200:
                print(f"Request {req_id} failed with status {resp.status}")
                return None
            
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        output_tokens += 1
                        if ttft is None:
                            t_first_token = time.perf_counter()
                            ttft = t_first_token - t_start
                except Exception:
                    pass
                    
        t_end = time.perf_counter()
        e2e = t_end - t_start
        
        tpot = None
        if output_tokens > 1 and t_first_token is not None:
            tpot = (t_end - t_first_token) / (output_tokens - 1)
            
        return {
            "req_id": req_id,
            "ttft_ms": ttft * 1000 if ttft else None,
            "tpot_ms": tpot * 1000 if tpot else None,
            "e2e_s": e2e,
            "tokens": output_tokens
        }
    except Exception as e:
        print(f"Error on request {req_id}: {e}")
        return None

async def worker(queue, session, results):
    while True:
        try:
            req_id, prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        res = await send_streaming_request(session, req_id, prompt)
        if res:
            results.append(res)
            print(f"[{len(results)}/{TOTAL_REQUESTS}] Req {req_id} done: {res['tokens']} tok, TTFT={res['ttft_ms']:.1f}ms, TPOT={res['tpot_ms']:.1f}ms")
        queue.task_done()

async def main():
    print(f"=== Starting Load Test: {TOTAL_REQUESTS} requests, concurrency={CONCURRENCY} ===")
    queue = asyncio.Queue()
    for i in range(TOTAL_REQUESTS):
        prompt = PROMPTS[i % len(PROMPTS)]
        queue.put_nowait((i + 1, prompt))
        
    results = []
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    t_bench_start = time.perf_counter()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [asyncio.create_task(worker(queue, session, results)) for _ in range(CONCURRENCY)]
        await asyncio.gather(*workers)
        
    t_bench_end = time.perf_counter()
    total_time = t_bench_end - t_bench_start
    
    valid_ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    valid_tpot = [r["tpot_ms"] for r in results if r["tpot_ms"] is not None]
    valid_e2e = [r["e2e_s"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)
    
    print("\n" + "="*50)
    print("           CLIENT-SIDE BENCHMARK RESULTS")
    print("="*50)
    print(f"Completed Requests:   {len(results)} / {TOTAL_REQUESTS}")
    print(f"Total Benchmark Time: {total_time:.2f} s")
    print(f"Total Tokens Output:  {total_tokens}")
    print(f"System Throughput:    {total_tokens / total_time:.1f} tokens/sec")
    print(f"Request Throughput:   {len(results) / total_time:.2f} req/sec")
    print("-" * 50)
    if valid_ttft:
        valid_ttft.sort()
        p50_ttft = statistics.median(valid_ttft)
        p90_ttft = valid_ttft[int(len(valid_ttft) * 0.90)]
        p99_ttft = valid_ttft[min(int(len(valid_ttft) * 0.99), len(valid_ttft) - 1)]
        print(f"TTFT  (p50 / p90 / p99): {p50_ttft:.1f}ms / {p90_ttft:.1f}ms / {p99_ttft:.1f}ms")
    if valid_tpot:
        valid_tpot.sort()
        p50_tpot = statistics.median(valid_tpot)
        p90_tpot = valid_tpot[int(len(valid_tpot) * 0.90)]
        p99_tpot = valid_tpot[min(int(len(valid_tpot) * 0.99), len(valid_tpot) - 1)]
        print(f"TPOT  (p50 / p90 / p99): {p50_tpot:.1f}ms / {p90_tpot:.1f}ms / {p99_tpot:.1f}ms")
    if valid_e2e:
        valid_e2e.sort()
        p50_e2e = statistics.median(valid_e2e)
        p99_e2e = valid_e2e[min(int(len(valid_e2e) * 0.99), len(valid_e2e) - 1)]
        print(f"E2E   (p50 / p99):       {p50_e2e:.2f}s / {p99_e2e:.2f}s")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())