import asyncio
import time
import json
import re
import statistics
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
CONCURRENCY = 10

TEST_SUITE = [
    # --- Category 1: Math & Reasoning (Exact Match on Final Number) ---
    {
        "id": "M1",
        "category": "Math & Reasoning",
        "prompt": "A farmer has 14 cows and 28 sheep. He sells half of his sheep and buys 5 more cows. How many total animals does he have now? End your answer with '#### <number>'.",
        "max_tokens": 120,
        "eval_type": "math",
        "gold": 33
    },
    {
        "id": "M2",
        "category": "Math & Reasoning",
        "prompt": "A train travels at 60 mph for 2.5 hours. It then stops for 30 minutes, and continues at 80 mph for 1.5 hours. What is the total distance traveled in miles? End your answer with '#### <number>'.",
        "max_tokens": 120,
        "eval_type": "math",
        "gold": 270
    },
    {
        "id": "M3",
        "category": "Math & Reasoning",
        "prompt": "A bakery bakes 120 cookies and packs them into boxes of 8. How many full boxes can they pack? End your answer with '#### <number>'.",
        "max_tokens": 80,
        "eval_type": "math",
        "gold": 15
    },
    {
        "id": "M4",
        "category": "Math & Reasoning",
        "prompt": "A shirt originally costs $80 and is discounted by 25 percent. What is the final price in dollars? End your answer with '#### <number>'.",
        "max_tokens": 80,
        "eval_type": "math",
        "gold": 60
    },
    {
        "id": "M5",
        "category": "Math & Reasoning",
        "prompt": "Lisa has twice as many apples as Tom. Tom has 6 apples. Lisa eats 3 apples and gives 4 to a friend. How many apples does Lisa have left? End your answer with '#### <number>'.",
        "max_tokens": 100,
        "eval_type": "math",
        "gold": 5
    },

    # --- Category 2: Factual Knowledge (Token F1 / Key Entity Match) ---
    {
        "id": "F1",
        "category": "Factual Knowledge",
        "prompt": "What is the capital city of Australia? Answer in one sentence.",
        "max_tokens": 30,
        "eval_type": "contains_all",
        "gold": ["canberra"]
    },
    {
        "id": "F2",
        "category": "Factual Knowledge",
        "prompt": "What is the chemical element with atomic number 1 on the periodic table?",
        "max_tokens": 30,
        "eval_type": "contains_all",
        "gold": ["hydrogen"]
    },
    {
        "id": "F3",
        "category": "Factual Knowledge",
        "prompt": "Who painted the Mona Lisa?",
        "max_tokens": 30,
        "eval_type": "contains_all",
        "gold": ["leonardo", "vinci"]
    },
    {
        "id": "F4",
        "category": "Factual Knowledge",
        "prompt": "What is the largest planet in our solar system?",
        "max_tokens": 30,
        "eval_type": "contains_all",
        "gold": ["jupiter"]
    },
    {
        "id": "F5",
        "category": "Factual Knowledge",
        "prompt": "Which organ in the human body is responsible for producing insulin?",
        "max_tokens": 30,
        "eval_type": "contains_all",
        "gold": ["pancreas"]
    },

    # --- Category 3: Instruction & Format Constraint Adherence ---
    {
        "id": "C1",
        "category": "Format & Constraints",
        "prompt": "Return ONLY a valid JSON object with keys 'status' (value 'success') and 'code' (value 200). Do not include any explanation or markdown formatting.",
        "max_tokens": 60,
        "eval_type": "json_keys",
        "gold": {"status": "success", "code": 200}
    },
    {
        "id": "C2",
        "category": "Format & Constraints",
        "prompt": "What is the color of a ripe banana? Respond using exactly ONE word and nothing else.",
        "max_tokens": 20,
        "eval_type": "single_word",
        "gold": ["yellow"]
    },
    {
        "id": "C3",
        "category": "Format & Constraints",
        "prompt": "Write a short sentence about birds flying. You must NOT use the letter 'e' anywhere in your sentence.",
        "max_tokens": 40,
        "eval_type": "no_letter_e",
        "gold": "e"
    },
    {
        "id": "C4",
        "category": "Format & Constraints",
        "prompt": "Output a raw JSON array containing exactly five numbers: 1, 2, 3, 4, 5. Output raw JSON only.",
        "max_tokens": 40,
        "eval_type": "json_array",
        "gold": [1, 2, 3, 4, 5]
    },
    {
        "id": "C5",
        "category": "Format & Constraints",
        "prompt": "List the three primary colors of pigment as a comma-separated list. No explanations.",
        "max_tokens": 30,
        "eval_type": "contains_any_set",
        "gold": [["red", "blue", "yellow"], ["cyan", "magenta", "yellow"]]
    },

    # --- Category 4: Context Extraction & Faithfulness (RAG) ---
    {
        "id": "R1",
        "category": "Context Faithfulness",
        "prompt": "CONTEXT: Project Apollo was founded in 2024 with a budget of 45 million credits. Dr. Elena Rostova is the lead director, operating out of the Elysium Crater on Mars. The launch date is fixed for October 12, 2027.\n\nQUESTION: According to the text, what is the allocated budget for Project Apollo?",
        "max_tokens": 40,
        "eval_type": "contains_all",
        "gold": ["45", "million"]
    },
    {
        "id": "R2",
        "category": "Context Faithfulness",
        "prompt": "CONTEXT: Project Apollo was founded in 2024 with a budget of 45 million credits. Dr. Elena Rostova is the lead director, operating out of the Elysium Crater on Mars. The launch date is fixed for October 12, 2027.\n\nQUESTION: According to the text, who is the lead director?",
        "max_tokens": 40,
        "eval_type": "contains_all",
        "gold": ["elena", "rostova"]
    },
    {
        "id": "R3",
        "category": "Context Faithfulness",
        "prompt": "CONTEXT: Project Apollo was founded in 2024 with a budget of 45 million credits. Dr. Elena Rostova is the lead director, operating out of the Elysium Crater on Mars. The launch date is fixed for October 12, 2027.\n\nQUESTION: According to the text, in which crater is the base located?",
        "max_tokens": 40,
        "eval_type": "contains_all",
        "gold": ["elysium"]
    },
    {
        "id": "R4",
        "category": "Context Faithfulness",
        "prompt": "CONTEXT: Project Apollo was founded in 2024 with a budget of 45 million credits. Dr. Elena Rostova is the lead director, operating out of the Elysium Crater on Mars. The launch date is fixed for October 12, 2027.\n\nQUESTION: According to the text, what is the scheduled launch date?",
        "max_tokens": 40,
        "eval_type": "contains_all",
        "gold": ["october", "12", "2027"]
    },
    {
        "id": "R5",
        "category": "Context Faithfulness",
        "prompt": "CONTEXT: Project Apollo was founded in 2024 with a budget of 45 million credits. Dr. Elena Rostova is the lead director, operating out of the Elysium Crater on Mars. The launch date is fixed for October 12, 2027.\n\nQUESTION: According to the text, in what year was the project founded?",
        "max_tokens": 40,
        "eval_type": "contains_all",
        "gold": ["2024"]
    }
]

def evaluate_response(item, text):
    t = text.strip().lower()
    etype = item["eval_type"]
    gold = item["gold"]
    
    if etype == "math":
        # Check #### <number> first, otherwise last number in text
        match = re.search(r"####\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            pred_num = float(match.group(1))
            return 1.0 if abs(pred_num - float(gold)) < 0.01 else 0.0
        # fallback: find all numbers, check last
        nums = re.findall(r"\b\d+\b", text)
        if nums and int(nums[-1]) == int(gold):
            return 1.0
        return 0.0
        
    elif etype == "contains_all":
        return 1.0 if all(word in t for word in gold) else 0.0
        
    elif etype == "contains_any_set":
        for s in gold:
            if all(word in t for word in s):
                return 1.0
        return 0.0
        
    elif etype == "single_word":
        words = re.findall(r"\b\w+\b", t)
        if len(words) == 1 and any(g in words[0] for g in gold):
            return 1.0
        return 0.0
        
    elif etype == "no_letter_e":
        cleaned = text.strip()
        if len(cleaned) > 5 and "e" not in cleaned.lower():
            return 1.0
        return 0.0
        
    elif etype == "json_keys":
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and parsed.get("status") == "success" and parsed.get("code") == 200:
                return 1.0
        except Exception:
            pass
        return 0.0
        
    elif etype == "json_array":
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and parsed == gold:
                return 1.0
        except Exception:
            pass
        return 0.0
        
    return 0.0

async def query_model(session, item, model_name):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": item["max_tokens"],
        "stream": True,
        "temperature": 0.0
    }
    
    t_start = time.perf_counter()
    ttft = None
    output_tokens = 0
    t_first = None
    accumulated_text = []
    
    try:
        async with session.post(URL, json=payload) as resp:
            if resp.status != 200:
                return None
            async for line in resp.content:
                l = line.decode("utf-8").strip()
                if not l or not l.startswith("data: "):
                    continue
                d = l[6:]
                if d == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        output_tokens += 1
                        accumulated_text.append(delta)
                        if ttft is None:
                            t_first = time.perf_counter()
                            ttft = t_first - t_start
                except Exception:
                    pass
        t_end = time.perf_counter()
        full_text = "".join(accumulated_text)
        tpot = (t_end - t_first) / (output_tokens - 1) if output_tokens > 1 and t_first else 0.0
        score = evaluate_response(item, full_text)
        
        return {
            "id": item["id"],
            "category": item["category"],
            "score": score,
            "response": full_text.strip().replace("\n", " ")[:70],
            "ttft_ms": ttft * 1000 if ttft else 0.0,
            "tpot_ms": tpot * 1000 if tpot else 0.0,
            "e2e_s": t_end - t_start,
            "tokens": output_tokens
        }
    except Exception as e:
        return None

async def run_benchmark(model_name):
    print(f"\n=========================================================================")
    print(f"      ALL-INCLUSIVE ACCURACY & PERFORMANCE EVALUATION: {model_name}")
    print(f"=========================================================================\n")
    
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    t0 = time.perf_counter()
    results = []
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [query_model(session, item, model_name) for item in TEST_SUITE]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r]
        
    t_total = time.perf_counter() - t0
    total_tokens = sum(r["tokens"] for r in results)
    
    print(f"{'ID':<4} | {'Category':<22} | {'Score':<5} | {'TTFT(ms)':<8} | {'TPOT(ms)':<8} | Sample Output")
    print("-" * 85)
    for r in results:
        status = " PASS" if r["score"] == 1.0 else " FAIL"
        print(f"{r['id']:<4} | {r['category']:<22} | {status} | {r['ttft_ms']:>8.1f} | {r['tpot_ms']:>8.1f} | {r['response']}")
        
    print("\n" + "="*70)
    print("                    ACCURACY & QUALITY SCORECARD")
    print("="*70)
    
    categories = ["Math & Reasoning", "Factual Knowledge", "Format & Constraints", "Context Faithfulness"]
    cat_scores = {}
    for c in categories:
        sub = [r for r in results if r["category"] == c]
        if sub:
            avg_score = (sum(r["score"] for r in sub) / len(sub)) * 100
            cat_scores[c] = avg_score
            print(f"  * {c:<25}: {avg_score:>5.1f}% accuracy  ({sum(1 for r in sub if r['score'] == 1.0)}/{len(sub)} passed)")
            
    composite_acc = statistics.mean(cat_scores.values()) if cat_scores else 0.0
    print("-" * 70)
    print(f"  >>> OVERALL COMPOSITE ACCURACY:  {composite_acc:.1f}% <<<")
    print("="*70)
    
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in results if r["tpot_ms"] > 0]
    ttfts.sort()
    tpots.sort()
    
    print("\n" + "="*70)
    print("                    PERFORMANCE SCORECARD")
    print("="*70)
    print(f"  * Total Evaluation Time:         {t_total:.2f} s")
    print(f"  * Total Tokens Generated:        {total_tokens} tokens")
    print(f"  * Overall Throughput:            {total_tokens / t_total:.1f} tokens/sec")
    print(f"  * TTFT Median (p50):             {statistics.median(ttfts):.1f} ms")
    print(f"  * TPOT Median (p50):             {statistics.median(tpots):.1f} ms/token")
    print(f"  * Production Selection Index:    {composite_acc / statistics.median(tpots):.2f} (Quality% / TPOT ms)")
    print("="*70)

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "tinyllama"
    asyncio.run(run_benchmark(model))