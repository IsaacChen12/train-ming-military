"""
Ollama Local Model Benchmark
Benchmarks all local models using the Ollama API and generates a comparison report.
Usage: python benchmark/run_benchmark.py
"""
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import box

OLLAMA_URL = "http://localhost:11434"
RAW_DIR = Path("benchmark/raw")
REPORT_PATH = Path("benchmark/report.md")

# Benchmark prompts (short/medium/long to test different load patterns)
PROMPTS = [
    {
        "id": "short",
        "label": "Short (math)",
        "text": "What is 15 multiplied by 37? Show the calculation."
    },
    {
        "id": "medium",
        "label": "Medium (history)",
        "text": "Describe the Ming Dynasty military equipment in 5 sentences."
    },
    {
        "id": "long",
        "label": "Long (code)",
        "text": "Write a Python function that implements binary search on a sorted list with proper error handling and docstring."
    },
]

SKIP_MODELS = {"kimi-k2.5:cloud"}  # cloud-only, no local weights

console = Console(highlight=False, markup=True)


def get_local_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return [m for m in models if m not in SKIP_MODELS]
    except Exception as e:
        console.print(f"[red]Cannot connect to Ollama: {e}[/red]")
        return []


def run_single(model: str, prompt: str, timeout: int = 300) -> dict:
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        wall_time = time.time() - start

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 1)
        load_duration_ns = data.get("load_duration", 0)
        total_duration_ns = data.get("total_duration", 0)

        gen_speed = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0
        prompt_speed = prompt_eval_count / (prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns else 0

        return {
            "status": "ok",
            "response_preview": data.get("response", "")[:200],
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "gen_speed_tps": round(gen_speed, 2),
            "prompt_speed_tps": round(prompt_speed, 2),
            "load_duration_s": round(load_duration_ns / 1e9, 3),
            "eval_duration_s": round(eval_duration_ns / 1e9, 3),
            "total_duration_s": round(total_duration_ns / 1e9, 3),
            "wall_time_s": round(wall_time, 3),
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "wall_time_s": timeout}
    except Exception as e:
        return {"status": "error", "error": str(e), "wall_time_s": round(time.time() - start, 3)}


def benchmark_model(model: str) -> dict:
    results = []
    for p in PROMPTS:
        console.print(f"    [{p['id']}] {p['label']}...", end=" ")
        result = run_single(model, p["text"])
        result["prompt_id"] = p["id"]
        result["prompt_label"] = p["label"]
        results.append(result)
        if result["status"] == "ok":
            console.print(f"[green]{result['gen_speed_tps']} tok/s[/green]")
        else:
            console.print(f"[red]{result['status']}[/red]")

    # Aggregate averages (only from successful runs)
    ok = [r for r in results if r["status"] == "ok"]
    avg_gen = round(sum(r["gen_speed_tps"] for r in ok) / len(ok), 2) if ok else 0
    avg_prompt = round(sum(r["prompt_speed_tps"] for r in ok) / len(ok), 2) if ok else 0
    avg_total = round(sum(r["total_duration_s"] for r in ok) / len(ok), 3) if ok else 0

    return {
        "model": model,
        "runs": results,
        "avg_gen_speed_tps": avg_gen,
        "avg_prompt_speed_tps": avg_prompt,
        "avg_total_duration_s": avg_total,
        "success_count": len(ok),
        "total_count": len(results),
    }


def generate_report(all_results: list):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Ollama Local Model Benchmark Report",
        f"\n**Generated:** {ts}  ",
        f"**Ollama URL:** {OLLAMA_URL}  ",
        f"**Models tested:** {len(all_results)}  ",
        f"**Prompts per model:** {len(PROMPTS)}  ",
        "\n---\n",
        "## Summary Table\n",
        "| Rank | Model | Avg Gen Speed (tok/s) | Avg Prompt Speed (tok/s) | Avg Total Time (s) | Success |",
        "|------|-------|-----------------------|--------------------------|-------------------|---------|",
    ]

    # Sort by generation speed descending
    sorted_results = sorted(all_results, key=lambda x: x["avg_gen_speed_tps"], reverse=True)

    for rank, r in enumerate(sorted_results, 1):
        success_str = f"{r['success_count']}/{r['total_count']}"
        lines.append(
            f"| {rank} | `{r['model']}` | **{r['avg_gen_speed_tps']}** | "
            f"{r['avg_prompt_speed_tps']} | {r['avg_total_duration_s']} | {success_str} |"
        )

    lines += [
        "\n---\n",
        "## Per-Prompt Breakdown\n",
    ]

    for r in sorted_results:
        lines.append(f"### `{r['model']}`\n")
        lines.append("| Prompt | Gen Speed (tok/s) | Prompt Speed (tok/s) | Gen Tokens | Total Time (s) | Status |")
        lines.append("|--------|-------------------|----------------------|------------|----------------|--------|")
        for run in r["runs"]:
            if run["status"] == "ok":
                lines.append(
                    f"| {run['prompt_label']} | {run['gen_speed_tps']} | "
                    f"{run['prompt_speed_tps']} | {run['eval_count']} | "
                    f"{run['total_duration_s']} | ✅ |"
                )
            else:
                lines.append(f"| {run['prompt_label']} | — | — | — | — | ❌ {run['status']} |")
        lines.append("")

    lines += [
        "---\n",
        "## Benchmark Prompts Used\n",
    ]
    for p in PROMPTS:
        lines.append(f"**{p['label']}:** {p['text']}\n")

    lines += [
        "---\n",
        "> Generated by `benchmark/run_benchmark.py`",
    ]

    return "\n".join(lines)


def print_summary_table(all_results: list):
    table = Table(title="Benchmark Results", box=box.ROUNDED)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Model", style="bold cyan")
    table.add_column("Avg Gen (tok/s)", justify="right", style="green")
    table.add_column("Avg Prompt (tok/s)", justify="right")
    table.add_column("Avg Total (s)", justify="right")
    table.add_column("OK/Total", justify="center")

    sorted_results = sorted(all_results, key=lambda x: x["avg_gen_speed_tps"], reverse=True)
    for rank, r in enumerate(sorted_results, 1):
        table.add_row(
            str(rank),
            r["model"],
            str(r["avg_gen_speed_tps"]),
            str(r["avg_prompt_speed_tps"]),
            str(r["avg_total_duration_s"]),
            f"{r['success_count']}/{r['total_count']}",
        )
    console.print(table)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Ollama Local Model Benchmark[/bold blue]")
    console.print(f"Connecting to {OLLAMA_URL}...\n")

    models = get_local_models()
    if not models:
        console.print("[red]No local models found.[/red]")
        return

    console.print(f"Found [bold]{len(models)}[/bold] local models: {', '.join(models)}")
    console.print(f"Skipped (cloud): {', '.join(SKIP_MODELS)}\n")

    all_results = []

    for model in models:
        console.print(f"\n[bold yellow]>> Benchmarking:[/bold yellow] {model}")
        result = benchmark_model(model)
        all_results.append(result)

        # Save raw result
        raw_path = RAW_DIR / f"{model.replace(':', '_').replace('/', '_')}.json"
        raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"  → Raw saved: {raw_path}")

    # Save all results combined
    combined_path = RAW_DIR / "_all_results.json"
    combined_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate markdown report
    report_md = generate_report(all_results)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    console.print("\n" + "=" * 60)
    print_summary_table(all_results)
    console.print(f"\n✅ Report saved: [bold]{REPORT_PATH}[/bold]")
    console.print(f"✅ Raw data:     [bold]{RAW_DIR}[/bold]")


if __name__ == "__main__":
    main()
