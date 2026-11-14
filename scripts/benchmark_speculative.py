import time
import requests
import json
import statistics

PROMPTS = [
    ("Code Generation", "Write a complete Python implementation of a Trie (Prefix Tree) data structure with insert, search, and startsWith methods. Include type hints and clean docstrings."),
    ("Technical Explanation", "Explain how the Raft consensus algorithm handles leader election and log replication during a network partition. Be thorough and technical."),
    ("Physics / Astronomy", "Describe the life cycle of a massive star from main sequence to core-collapse supernova and neutron star formation. Detail the fusion stages."),
    ("Historical Summary", "Provide a detailed overview of the Industrial Revolution, focusing on the transition from agrarian economies to steam-powered manufacturing."),
    ("System Design", "Design a distributed rate limiter supporting millions of requests per second. Explain token bucket vs leaky bucket algorithms and Redis storage.")
]

def get_spec_metrics():
    try:
        r = requests.get("http://localhost:8000/metrics", timeout=3)
        drafted = 0
        accepted = 0
        for line in r.text.splitlines():
            if "vllm:spec_decode_draft_tokens_total" in line and not line.startswith("#"):
                drafted = float(line.split()[-1])
            elif "vllm:spec_decode_accepted_tokens_total" in line and not line.startswith("#"):
                accepted = float(line.split()[-1])
        rate = (accepted / drafted * 100) if drafted > 0 else 0
        return drafted, accepted, rate
    except Exception:
        return 0, 0, 0

def run_benchmark():
    print(f"\n=========================================================================")
    print(f"      SPECULATIVE DECODING BENCHMARK: Qwen-1.5B (Draft: Qwen-0.5B)")
    print(f"=========================================================================\n")
    print(f"{'Task':<22} | {'Tokens':<6} | {'E2E (s)':<8} | {'TPOT (ms)':<10} | {'Speed (tok/s)':<14} | {'Sample Output'}")
    print("-" * 88)

    tpots = []
    speeds = []

    for category, prompt in PROMPTS:
        payload = {
            "model": "qwen-1.5b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 250,
            "stream": True
        }

        t0 = time.perf_counter()
        r = requests.post("http://localhost:8000/v1/chat/completions", json=payload, stream=True)
        
        t_first = None
        output_tokens = 0
        text = ""

        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if t_first is None:
                            t_first = time.perf_counter()
                        output_tokens += 1
                        text += delta
                except Exception:
                    pass

        t_end = time.perf_counter()
        
        gen_duration = t_end - t_first if (t_first and output_tokens > 1) else 0.001
        tpot_ms = (gen_duration / max(1, (output_tokens - 1))) * 1000
        speed_tok_s = (output_tokens - 1) / gen_duration
        e2e_s = t_end - t0

        tpots.append(tpot_ms)
        speeds.append(speed_tok_s)

        print(f"{category:<22} | {output_tokens:<6} | {e2e_s:7.2f}s | {tpot_ms:9.2f} | {speed_tok_s:12.1f} | {text.strip().replace(chr(10), ' ')[:30]}...")

    avg_tpot = statistics.mean(tpots)
    median_tpot = statistics.median(tpots)
    avg_speed = statistics.mean(speeds)
    drafted, accepted, acc_rate = get_spec_metrics()

    print(f"\n=======================================================================")
    print(f"                      SPECULATIVE DECODING SUMMARY")
    print(f"=======================================================================")
    print(f"  * Median Generation Latency (TPOT) : {median_tpot:6.2f} ms/token")
    print(f"  * Average Generation Speed         : {avg_speed:6.1f} tokens/sec")
    print(f"  * Draft Tokens Proposed            : {drafted:,.0f}")
    print(f"  * Draft Tokens Accepted by 1.5B    : {accepted:,.0f}")
    print(f"  * Speculative Acceptance Rate      : {acc_rate:6.1f} %")
    print(f"=======================================================================\n")

if __name__ == '__main__':
    run_benchmark()
