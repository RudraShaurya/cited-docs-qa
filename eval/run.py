"""Rerunnable evaluation harness.

Usage:
    python -m eval.run --mode hybrid --chunk 450 --out eval/results/hybrid_450.json
    python -m eval.run --mode vector --retrieval-only
    python -m eval.run --mode hybrid --sample-correctness 40 --out eval/results/sample.jsonl

Every number quoted in the README/EVALUATION.md must come from a JSON file
this script wrote - never a hand-typed estimate.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app.answer import answer_question
from app.llm.factory import get_llm_client
from app.retrieval.hybrid import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ANSWERABLE_TYPES = {"answerable_single", "answerable_multi"}


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _matches_gold(chunk: dict, gold_item: dict) -> bool:
    return chunk.get("title") == gold_item.get("title") and chunk.get("page") == gold_item.get("page")


def retrieval_metrics(questions: list[dict], mode: str, k: int = 5, mrr_k: int = 10) -> dict:
    answerable = [q for q in questions if q.get("type") in ANSWERABLE_TYPES]

    total_gold = 0
    found_at_k = 0
    reciprocal_ranks = []

    for q in answerable:
        gold_items = q.get("gold", [])
        total_gold += len(gold_items)

        results = retrieve(q["question"], k=mrr_k, mode=mode)

        for gold_item in gold_items:
            if any(_matches_gold(c, gold_item) for c in results[:k]):
                found_at_k += 1

        rank = next(
            (i + 1 for i, c in enumerate(results) if any(_matches_gold(c, g) for g in gold_items)),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    recall_at_k = found_at_k / total_gold if total_gold else None
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else None

    return {
        "recall_at_k": recall_at_k,
        "k": k,
        "mrr_at_k": mrr,
        "mrr_k": mrr_k,
        "answerable_questions": len(answerable),
        "total_gold_items": total_gold,
    }


def refusal_and_latency_metrics(questions: list[dict], mode: str) -> dict:
    llm = get_llm_client()

    unanswerable = [q for q in questions if q.get("type") == "unanswerable"]
    answerable = [q for q in questions if q.get("type") in ANSWERABLE_TYPES]

    correct_refusals = 0
    wrong_refusals = 0
    retrieve_ms, generate_ms, total_ms = [], [], []

    for q in unanswerable:
        result = answer_question(q["question"], llm, mode=mode)
        if result.refused:
            correct_refusals += 1
        retrieve_ms.append(result.timings_ms.get("retrieve", 0.0))
        generate_ms.append(result.timings_ms.get("generate", 0.0))
        total_ms.append(result.timings_ms.get("total", 0.0))

    for q in answerable:
        result = answer_question(q["question"], llm, mode=mode)
        if result.refused:
            wrong_refusals += 1
        retrieve_ms.append(result.timings_ms.get("retrieve", 0.0))
        generate_ms.append(result.timings_ms.get("generate", 0.0))
        total_ms.append(result.timings_ms.get("total", 0.0))

    def pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        idx = min(len(values) - 1, int(round(p * (len(values) - 1))))
        return values[idx]

    return {
        "unanswerable_questions": len(unanswerable),
        "correct_refusal_rate": correct_refusals / len(unanswerable) if unanswerable else None,
        "answerable_questions": len(answerable),
        "wrong_refusal_rate": wrong_refusals / len(answerable) if answerable else None,
        "latency_ms": {
            "retrieve_p50": pct(retrieve_ms, 0.5),
            "retrieve_p95": pct(retrieve_ms, 0.95),
            "generate_p50": pct(generate_ms, 0.5),
            "generate_p95": pct(generate_ms, 0.95),
            "total_p50": pct(total_ms, 0.5),
            "total_p95": pct(total_ms, 0.95),
        },
        "model": getattr(llm, "model", None),
    }


def write_hand_scoring_sample(questions: list[dict], mode: str, n: int, out_path: Path) -> None:
    """Dump N answerable Q&A pairs for manual 0/1/2 correctness + faithfulness scoring."""
    llm = get_llm_client()
    answerable = [q for q in questions if q.get("type") in ANSWERABLE_TYPES][:n]

    rows = []
    for q in answerable:
        result = answer_question(q["question"], llm, mode=mode)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "reference": q.get("reference"),
                "answer": result.answer,
                "citations": [vars(c) for c in result.citations],
                "correctness_0_1_2": None,
                "faithful_yes_no": None,
                "notes": "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Skip LLM calls; only compute Recall/MRR."
    )
    parser.add_argument(
        "--sample-correctness",
        type=int,
        default=0,
        help="Instead of full metrics, dump N answers to a hand-scoring JSONL file.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)

    if args.sample_correctness:
        out_path = args.out or RESULTS_DIR / f"hand_scoring_{args.mode}.jsonl"
        write_hand_scoring_sample(questions, args.mode, args.sample_correctness, out_path)
        print(f"Wrote {args.sample_correctness} rows to {out_path} for hand scoring.")
        return

    report = {
        "mode": args.mode,
        "k": args.k,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "retrieval": retrieval_metrics(questions, args.mode, k=args.k),
    }

    if not args.retrieval_only:
        report["generation"] = refusal_and_latency_metrics(questions, args.mode)

    out_path = args.out or RESULTS_DIR / f"{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
