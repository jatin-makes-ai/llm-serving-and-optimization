import asyncio
import time
import json
import random
import statistics
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "tinyllama"
CONCURRENCY = 30

SHORT_TOPICS = [
    "What is the boiling point of water at high altitude?",
    "Convert 45 Celsius to Fahrenheit.",
    "Give me 3 synonyms for the word 'resilient'.",
    "What year was the Python programming language first released?",
    "Explain the word 'serendipity' in one sentence.",
    "What is the chemical symbol for Tungsten?",
    "Who directed the movie Interstellar?",
    "How many bones are in the adult human body?",
    "What is the speed of light in vacuum?",
    "Name the capital city of Australia.",
    "What does HTTP status code 404 mean?",
    "How do you declare a constant in modern JavaScript?",
    "What is the deepest ocean trench on Earth?",
    "What does CPU stand for?",
    "Who developed the theory of general relativity?"
]

MEDIUM_TOPICS = [
    "Draft a polite 3-paragraph email to a vendor requesting a 10% discount on an annual enterprise license renewal.",
    "Write a clean Python function that performs binary search on a sorted list of numbers and returns the index or -1.",
    "Explain the architectural differences between SQL (relational) and NoSQL (document) databases, with pros and cons for each.",
    "Describe the key factors that cause seasonal monsoons in South Asia and how agricultural economies rely on them.",
    "Write a concise code review comment pointing out why using a bare 'except:' clause in Python is bad practice.",
    "Summarize the plot and philosophical themes of George Orwell's novel '1984' in two structured paragraphs.",
    "Explain the difference between concurrency and parallelism using an intuitive real-world coffee shop analogy.",
    "Write a Bash one-liner to find all .log files in /var/log modified within the last 24 hours and compress them."
]

LONG_ARTICLE_TEMPLATE = """You are an expert financial and technological research analyst. Carefully read the following detailed background report and provide a comprehensive analysis:

REPORT CONTEXT:
The semiconductor manufacturing landscape has undergone massive consolidation over the past two decades. As feature sizes shrunk from 90 nanometers down to sub-3 nanometer nodes utilizing Gate-All-Around (GAA) nanosheet transistors and High-NA Extreme Ultraviolet (EUV) lithography, the capital expenditure required to construct leading-edge fabrication facilities ('fabs') has escalated exponentially. A modern high-volume fab today demands upwards of twenty billion dollars in upfront investment.

This capital intensity has driven a bifurcation in the industry: fabless design companies (such as Nvidia, AMD, Apple, and Qualcomm) focus exclusively on microarchitecture, circuit design, and software toolchains, while outsourcing physical production to dedicated pure-play foundries, predominantly TSMC, with Intel Foundry and Samsung Electronics competing for second source market share. 

Simultaneously, the surge in Generative AI model training and inference workloads has generated unprecedented demand for high-bandwidth memory (HBM3e and HBM4), advanced 2.5D/3D packaging techniques (such as TSMC's CoWoS), and specialized silicon architectures optimized for tensor operations and low-precision floating point arithmetic (FP8, FP4). Geopolitical considerations, supply chain security, and sovereign semiconductor initiatives (such as the US CHIPS Act and the European Chips Act) are actively reshaping fab geographic distribution.

INSTRUCTIONS:
{prompt_instruction}"""

LONG_TASKS = [
    "Synthesize the report above into 3 core strategic risks facing fabless hardware companies over the next 5 years, and recommend 2 mitigation strategies for each.",
    "From the report above, analyze why packaging technologies like CoWoS have become as critical a bottleneck as silicon lithography in modern AI accelerators.",
    "Based on the context, evaluate the economic viability of new market entrants attempting to build greenfield semiconductor foundries today.",
    "Identify the primary technological hurdles mentioned in shrinking transistor sizes below 3 nanometers and explain their implications for server energy efficiency."
]

def generate_request_pool(total=90):
    pool = []
    
    # 30 Short Requests (Prompt ~20 tokens, Output max 40 tokens)
    for i in range(30):
        t = SHORT_TOPICS[i % len(SHORT_TOPICS)]
        pool.append({
            "type": "Short",
            "prompt": f"Answer concisely in one or two sentences: {t}",
            "max_tokens": 40
        })
        
    # 30 Medium Requests (Prompt ~150 tokens, Output max 180 tokens)
    for i in range(30):
        t = MEDIUM_TOPICS[i % len(MEDIUM_TOPICS)]
        pool.append({
            "type": "Medium",
            "prompt": f"Provide a detailed, practical response to the following prompt:\n\n{t}",
            "max_tokens": 180
        })
        
    # 30 Long Requests (Prompt ~650 tokens, Output max 350 tokens)
    for i in range(30):
        task = LONG_TASKS[i % len(LONG_TASKS)]
        prompt = LONG_ARTICLE_TEMPLATE.format(prompt_instruction=task)
        pool.append({
            "type": "Long (RAG)",
            "prompt": prompt,
            "max_tokens": 350
        })
        
    random.seed(42)
    random.shuffle(pool)
    return pool

async def send_streaming_request(session, req_id, item):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": item["max_tokens"],
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
                print(f"Req {req_id} failed with status {resp.status}")
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
            "type": item["type"],
            "ttft_ms": ttft * 1000 if ttft else None,
            "tpot_ms": tpot * 1000 if tpot else None,
            "e2e_s": e2e,
            "tokens": output_tokens
        }
    except Exception as e:
        print(f"Error on req {req_id}: {e}")
        return None

async def worker(queue, session, results, total_count):
    while True:
        try:
            req_id, item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        res = await send_streaming_request(session, req_id, item)
        if res:
            results.append(res)
            print(f"[{len(results):02d}/{total_count}] {res['type']:<10} | Out: {res['tokens']:>3} tok | TTFT: {res['ttft_ms']:>6.1f}ms | TPOT: {res['tpot_ms']:>5.1f}ms | E2E: {res['e2e_s']:>4.2f}s")
        queue.task_done()

async def main():
    pool = generate_request_pool(total=90)
    total_count = len(pool)
    print(f"=== Starting Realistic Multi-Tier Load Test ===")
    print(f"Total Requests: {total_count} (30 Short, 30 Medium, 30 Long RAG)")
    print(f"Concurrency:    {CONCURRENCY} parallel streams\n")
    
    queue = asyncio.Queue()
    for i, item in enumerate(pool):
        queue.put_nowait((i + 1, item))
        
    results = []
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    t_start = time.perf_counter()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [asyncio.create_task(worker(queue, session, results, total_count)) for _ in range(CONCURRENCY)]
        await asyncio.gather(*workers)
        
    t_end = time.perf_counter()
    total_time = t_end - t_start
    total_tokens = sum(r["tokens"] for r in results)
    
    print("\n" + "="*70)
    print("                REALISTIC LOAD TEST RESULTS")
    print("="*70)
    print(f"Completed Requests:   {len(results)} / {total_count}")
    print(f"Total Benchmark Time: {total_time:.2f} s")
    print(f"Total Output Tokens:  {total_tokens}")
    print(f"Overall Throughput:   {total_tokens / total_time:.1f} tokens/sec")
    print(f"Request Throughput:   {len(results) / total_time:.2f} req/sec")
    print("-" * 70)
    
    for category in ["Short", "Medium", "Long (RAG)"]:
        cat_res = [r for r in results if r["type"] == category]
        if not cat_res:
            continue
        ttfts = [r["ttft_ms"] for r in cat_res if r["ttft_ms"]]
        tpots = [r["tpot_ms"] for r in cat_res if r["tpot_ms"]]
        e2es = [r["e2e_s"] for r in cat_res]
        tokens = [r["tokens"] for r in cat_res]
        
        ttfts.sort()
        tpots.sort()
        e2es.sort()
        
        p50_ttft = statistics.median(ttfts)
        p99_ttft = ttfts[min(int(len(ttfts)*0.99), len(ttfts)-1)]
        p50_tpot = statistics.median(tpots)
        p50_e2e = statistics.median(e2es)
        avg_tokens = statistics.mean(tokens)
        
        print(f"[{category:<10}] ({len(cat_res)} reqs) Avg Tokens: {avg_tokens:.0f} | TTFT p50: {p50_ttft:.1f}ms, p99: {p99_ttft:.1f}ms | TPOT p50: {p50_tpot:.1f}ms | E2E p50: {p50_e2e:.2f}s")
        
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())