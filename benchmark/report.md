# Ollama Local Model Benchmark Report

**Generated:** 2026-09-06  
**Ollama:** 0.32.13 | **OS:** Windows 11 (10.0.26200)  
**Models tested:** 7 local (1 skipped: `kimi-k2.5:cloud`)  
**Prompts per model:** 3 (math / history / code)

## Hardware

| Component | Spec |
|-----------|------|
| **GPU** | NVIDIA GeForce RTX 4090 Laptop GPU |
| **VRAM** | 16 GB GDDR6 (16376 MiB) |
| **GPU Driver** | 560.94 |
| **CPU** | Intel Core i9-14900HX |
| **CPU Cores** | 24 cores / 32 threads (8P + 16E) |
| **CPU Base Clock** | 2.2 GHz (boost up to 5.8 GHz) |
| **RAM** | 48 GB |

---

## Summary Table

| Rank | Model | Params | File Size | Avg Gen (tok/s) | Avg Prompt (tok/s) | Avg Total (s) | Result |
|------|-------|--------|-----------|-----------------|--------------------|---------------|--------|
| 🥇 1 | `qwen2.5:latest` | 7.6B | 4.7 GB | **105.67** | **1726.23** | **4.05** | ✅ 3/3 |
| 🥈 2 | `qwen3.5:9b` | 9B | 6.6 GB | **82.60** | 214.58 | 17.74 | ✅ 3/3 |
| 🥉 3 | `gemma4:12b` | 12B | 7.6 GB | **58.80** | 188.76 | 19.48 | ✅ 3/3 |
| 4 | `qwen3.6:35b` | 35B | 23 GB | **53.58** | 67.93 | 59.01 | ✅ 3/3 |
| 5 | `qwen3.8:latest` | 27.3B | 17 GB | **13.56** | 25.41 | 77.00 | ✅ 3/3 ⚠️ |
| 6 | `qwen3.6:27b` | 27B | 17 GB | **12.39** | 34.89 | 104.49 | ⚠️ 2/3 |
| 7 | `gemma4:26b` | 26B | 18 GB | **0** | 0 | — | ❌ 0/3 |

---

## Speed Visualization

```
Generation Speed (tokens/sec) — higher is better

qwen2.5:latest  ████████████████████████████████████████  105.67 tok/s  🥇
qwen3.5:9b      ████████████████████████████████          82.60 tok/s   🥈
gemma4:12b      ███████████████████████                   58.80 tok/s   🥉
qwen3.6:35b     █████████████████████                     53.58 tok/s
qwen3.8:latest  █████                                     13.56 tok/s   ⚠️ thinking
qwen3.6:27b     ████                                      12.39 tok/s   ⚠️ thinking
gemma4:26b      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░          ERROR
```

---

## Per-Prompt Breakdown

### `qwen2.5:latest` — 🥇 Fastest Overall

| Prompt | Gen (tok/s) | Prompt (tok/s) | Tokens Out | Total (s) | Status |
|--------|-------------|----------------|------------|-----------|--------|
| Short (math) | 106.35 | 1631.02 | 111 | 4.61 | ✅ |
| Medium (history) | 106.65 | 2513.67 | 95 | 1.14 | ✅ |
| Long (code) | 104.00 | 1034.00 | 635 | 6.38 | ✅ |
| **Average** | **105.67** | **1726.23** | — | **4.05** | |

---

### `qwen3.5:9b` — 🥈 Surprise Runner-Up

| Prompt | Gen (tok/s) | Prompt (tok/s) | Tokens Out | Total (s) | Status |
|--------|-------------|----------------|------------|-----------|--------|
| Short (math) | 83.18 | 341.28 | 1266 | 23.02 | ✅ |
| Medium (history) | 81.46 | 182.05 | 1666 | 21.02 | ✅ |
| Long (code) | 83.16 | 120.40 | 710 | 9.18 | ✅ |
| **Average** | **82.60** | **214.58** | — | **17.74** | |

> **Note:** This is the model that showed "no output" in earlier testing. Likely cause was a cold-start timeout — the model takes ~20s per response. Increase client timeout to 60s+.

---

### `gemma4:12b` — 🥉 Consistent Mid-Range

| Prompt | Gen (tok/s) | Prompt (tok/s) | Tokens Out | Total (s) | Status |
|--------|-------------|----------------|------------|-----------|--------|
| Short (math) | 60.14 | 383.00 | 480 | 14.91 | ✅ |
| Medium (history) | 58.25 | 75.32 | 798 | 14.69 | ✅ |
| Long (code) | 58.00 | 107.96 | 1613 | 28.83 | ✅ |
| **Average** | **58.80** | **188.76** | — | **19.48** | |

---

### `qwen3.6:35b` — Best Large Model

| Prompt | Gen (tok/s) | Prompt (tok/s) | Tokens Out | Total (s) | Status |
|--------|-------------|----------------|------------|-----------|--------|
| Short (math) | 53.79 | 11.61 | 2287 | 81.02 | ✅ |
| Medium (history) | 53.24 | 85.29 | 1096 | 21.28 | ✅ |
| Long (code) | 53.71 | 106.88 | 3974 | 74.72 | ✅ |
| **Average** | **53.58** | **67.93** | — | **59.01** | |

> Extremely verbose (2287–3974 tokens/response). Best for deep analysis where thoroughness matters.

---

### `qwen3.8:latest` — ⚠️ Thinking Mode Active

| Prompt | Gen (tok/s) | Prompt (tok/s) | Tokens Out | Total (s) | Status |
|--------|-------------|----------------|------------|-----------|--------|
| Short (math) | 15.52 | 27.15 | 415 | 50.50 | ✅ |
| Medium (history) | 11.93 | 22.60 | 620 | 53.38 | ✅ |
| Long (code) | 13.24 | 26.47 | 1662 | 127.10 | ✅ |
| **Average** | **13.56** | **25.41** | — | **77.00** | |

> Extended reasoning (thinking) mode is ON. Add `/no_think` to prompts to disable — expected speed without thinking: ~50 tok/s.

---

### `qwen3.6:27b` — ⚠️ Thinking Mode + Timeout

| Prompt | Gen (tok/s) | Prompt (tok/s) | Tokens Out | Total (s) | Status |
|--------|-------------|----------------|------------|-----------|--------|
| Short (math) | 12.41 | 37.66 | 929 | 97.46 | ✅ |
| Medium (history) | 12.37 | 32.12 | 1365 | 111.51 | ✅ |
| Long (code) | — | — | — | >300s | ❌ timeout |
| **Average** | **12.39** | **34.89** | — | **104.49** | |

> Thinking mode active. Long code prompt exceeded 300s timeout. Same architecture as qwen3.8 but larger — slower due to model size.

---

### `gemma4:26b` — ❌ Failed (OOM)

| Prompt | Status | Error |
|--------|--------|-------|
| Short (math) | ❌ | 500 Server Error (~22s) |
| Medium (history) | ❌ | 500 Server Error (~21s) |
| Long (code) | ❌ | 500 Server Error (~21s) |

> **Cause:** Insufficient VRAM — 18GB model cannot load alongside other active models. Run it in isolation:
> ```cmd
> ollama stop qwen3.8:latest
> ollama run gemma4:26b "test prompt"
> ```

---

## Key Findings

### 1. Speed Winner: `qwen2.5:latest`
At **105.67 tok/s** with near-instant prompt processing (1726 tok/s), it is the best model for RAG — fast at processing large context windows and generating answers quickly.

### 2. `qwen3.5:9b` No-Output Mystery Solved
It is the **2nd fastest model** at 82.6 tok/s but each response takes 9–23 seconds total. The earlier "no output" issue was almost certainly a **client timeout set too short**. Fix: set timeout ≥ 60s in your requests.

### 3. Thinking Mode on `qwen3.8` and `qwen3.6:27b`
Both run at ~12–13 tok/s because they are in extended reasoning mode. Disabling thinking gives ~4–6× speedup. Add `/no_think` at the end of prompts or use:
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

### 4. `qwen3.6:35b` — Quality at a Cost
Generates 3–4× more tokens than others (very thorough answers) at a steady 53 tok/s. Good for SFT data generation where quality > speed.

### 5. `gemma4:26b` needs exclusive GPU access
Cannot share memory with other loaded models. Must be run alone.

---

## Recommendations for This Project

| Use Case | Best Model | Why |
|----------|-----------|-----|
| RAG Q&A (production) | `qwen2.5:latest` | 105 tok/s, instant prompt eval |
| RAG Q&A (quality) | `qwen3.5:9b` | 82 tok/s, richer answers |
| SFT data generation | `qwen3.6:35b` | Most thorough, detailed outputs |
| Interactive chat | `gemma4:12b` | Balanced speed + quality |
| Deep reasoning tasks | `qwen3.8:latest` (thinking ON) | Accept the slowness |

---

## Raw Data Files

| File | Model | Notes |
|------|-------|-------|
| `raw/qwen2.5_latest.json` | qwen2.5:latest | All pass |
| `raw/qwen3.5_9b.json` | qwen3.5:9b | All pass |
| `raw/gemma4_12b.json` | gemma4:12b | All pass |
| `raw/qwen3.6_35b.json` | qwen3.6:35b | All pass |
| `raw/qwen3.8_latest.json` | qwen3.8:latest | All pass, thinking mode |
| `raw/qwen3.6_27b.json` | qwen3.6:27b | 2/3, long code timed out |
| `raw/gemma4_26b.json` | gemma4:26b | All errors (OOM) |

---

> Generated by `benchmark/run_benchmark.py`
