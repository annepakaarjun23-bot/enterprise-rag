import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "240")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.test_case import LLMTestCase

from evals.adapters.retrieve_adapter import retrieve
from evals.dataset_io import load_dataset
from evals.judge import judge
from evals.metrics.e2e_metrics import get_e2e_metrics
from rag.generation.answerer import generate_answer

DATASET = REPO_ROOT / "evals" / "datasets" / "golden_e2e.jsonl"
K = 5
MODE = "hybrid"
AUTOMERGE = True
RERANK = False
GEN_CONCURRENCY = 4
MAX_CONCURRENT = 2
LIMIT = None

RESULTS_DIR = REPO_ROOT / "evals" / "results"
JUDGE_MODEL = judge.get_model_name()


def retrieve_contexts(items: list[dict]) -> list[list[str]]:
    return [
        [r.text for r in retrieve(query=item["input"], k=K, mode=MODE, automerge=AUTOMERGE, rerank=RERANK)]
        for item in items
    ]


async def generate_one(item: dict, context: list[str], sem: asyncio.Semaphore) -> str | None:
    async with sem:
        try:
            return await generate_answer(item["input"], context)
        except Exception as e:
            print(f"[generate failed] {item['id']}: {type(e).__name__}: {e}")
            return None


async def generate_all(items: list[dict], contexts: list[list[str]]) -> list[str | None]:
    sem = asyncio.Semaphore(GEN_CONCURRENCY)
    return await asyncio.gather(*(generate_one(i, c, sem) for i, c in zip(items, contexts)))


def build_cases(items: list[dict], contexts: list[list[str]], answers: list[str | None]) -> list[LLMTestCase]:
    return [
        LLMTestCase(
            name=item["id"],
            input=item["input"],
            actual_output=answer,
            expected_output=item["expected_output"],
            retrieval_context=context,
        )
        for item, context, answer in zip(items, contexts, answers)
        if answer is not None
    ]


def collect_rows(test_results) -> list[dict]:
    rows = []
    for tr in test_results:
        metrics = [
            {"name": m.name, "score": m.score, "passed": m.success, "error": m.error}
            for m in (tr.metrics_data or [])
        ]
        rows.append({
            "id": tr.name,
            "input": tr.input,
            "answer": tr.actual_output,
            "success": tr.success,
            "has_error": (not metrics) or any(m["error"] for m in metrics),
            "metrics": metrics,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    per_metric: dict[str, list] = {}
    for r in rows:
        for m in r["metrics"]:
            per_metric.setdefault(m["name"], []).append(m)

    metrics = {}
    for name, ms in per_metric.items():
        scores = [m["score"] for m in ms if m["score"] is not None]
        metrics[name] = {
            "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
            "passed": sum(1 for m in ms if m["passed"]),
            "total": len(ms),
        }
    return {
        "questions": len(rows),
        "passed_all_metrics": sum(1 for r in rows if r["success"]),
        "metrics": metrics,
        "errored_ids": [r["id"] for r in rows if r["has_error"]],
        "failed_ids": [r["id"] for r in rows if not r["success"]],
    }


def main() -> None:
    items = load_dataset(DATASET)
    for i, q in enumerate(items, 1):
        q.setdefault("id", q.get("qid") or f"row{i:03d}")
    if LIMIT:
        items = items[:LIMIT]
    print(f"[setup] {len(items)} questions | mode={MODE} automerge={AUTOMERGE} rerank={RERANK} k={K}")
    print(f"[setup] judge: {JUDGE_MODEL} | concurrency: {MAX_CONCURRENT}")

    t0 = time.time()
    contexts = retrieve_contexts(items)
    t_retrieve = time.time() - t0
    print(f"[retrieve] {t_retrieve:.1f}s total, {t_retrieve / max(len(items), 1):.2f}s per question")

    t1 = time.time()
    answers = asyncio.run(generate_all(items, contexts))
    t_generate = time.time() - t1
    cases = build_cases(items, contexts, answers)
    print(f"[generate] {len(cases)}/{len(items)} answers in {t_generate:.1f}s")

    t2 = time.time()
    res = evaluate(
        test_cases=cases,
        metrics=get_e2e_metrics(),
        async_config=AsyncConfig(max_concurrent=MAX_CONCURRENT, throttle_value=0),
        error_config=ErrorConfig(ignore_errors=True),
    )
    t_judge = time.time() - t2

    rows = collect_rows(res.test_results)
    summary = summarize(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"e2e_{MODE}_automerge{AUTOMERGE}_rerank{RERANK}_k{K}_{JUDGE_MODEL.split('/')[-1]}_{stamp}"
    config = {
        "mode": MODE, "automerge": AUTOMERGE, "rerank": RERANK, "k": K,
        "judge_model": JUDGE_MODEL, "max_concurrent": MAX_CONCURRENT,
        "gen_concurrency": GEN_CONCURRENCY,
    }
    timing = {
        "retrieve_s": round(t_retrieve, 1),
        "generate_s": round(t_generate, 1),
        "judge_s": round(t_judge, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_name}.json"
    out.write_text(
        json.dumps(
            {"run": run_name, "config": config, "timing": timing, "summary": summary, "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    flat = {
        "run": run_name,
        **config,
        "questions": summary["questions"],
        **{f"{n}_avg": v["avg_score"] for n, v in summary["metrics"].items()},
        **timing,
    }
    with (RESULTS_DIR / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(flat, ensure_ascii=False) + "\n")

    print(f"\n===== {run_name} =====")
    print(f"questions: {summary['questions']} | passed all metrics: {summary['passed_all_metrics']}")
    for name, v in summary["metrics"].items():
        print(f"  {name:<22} avg {v['avg_score']} | passed {v['passed']}/{v['total']}")
    if summary["errored_ids"]:
        print(f"\n{len(summary['errored_ids'])} question(s) had judge errors: {summary['errored_ids']}")
    print(
        f"time: retrieve {timedelta(seconds=int(t_retrieve))} + generate {timedelta(seconds=int(t_generate))}"
        f" + judge {timedelta(seconds=int(t_judge))}"
    )
    print(f"saved: {out}\n")


if __name__ == "__main__":
    main()