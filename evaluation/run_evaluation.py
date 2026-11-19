from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_system.config import settings
from rag_system.retrieval import RetrievalEngine
from rag_system.service import KnowledgeService, ABSTAIN_TEXT


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_coverage(results: list[dict], question: dict) -> tuple[float | None, float | None]:
    evidence = question["evidence"]
    def matches(item, expected):
        return (item["filename"] == expected["file"]
                and "".join(expected["term"].split()) in "".join(item["content"].split()))
    if not evidence:
        return None, None
    found = []
    for expected in evidence:
        found.append(any(
            matches(item, expected)
            for item in results
        ))
    ranks = []
    for expected in evidence:
        for rank, item in enumerate(results, 1):
            if matches(item, expected):
                ranks.append(1.0 / rank)
                break
        else:
            ranks.append(0.0)
    return sum(found) / len(found), sum(ranks) / len(ranks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible RAG evaluation set")
    parser.add_argument("--questions", type=Path, default=ROOT / "evaluation" / "questions.jsonl")
    parser.add_argument("--generate", action="store_true", help="also call the generation API for all questions")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation_results.json")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("The evaluation set is empty.")
    service = KnowledgeService()
    chunks = service.db.all_chunks()
    if not chunks:
        raise SystemExit("Knowledge base is empty. Start the app and import demo_data first.")
    engine = RetrievalEngine(chunks)
    outputs = {mode: [] for mode in ("vector", "bm25", "hybrid", "hybrid_rerank")}
    generation_results = []

    for idx, question in enumerate(questions, 1):
        started = time.perf_counter()
        vector = service.provider.embed_query(question["question"])
        embed_ms = int((time.perf_counter() - started) * 1000)
        for mode in outputs:
            tic = time.perf_counter()
            results = engine.retrieve(
                question["question"], vector, mode="hybrid" if mode == "hybrid_rerank" else mode,
                rerank=(mode == "hybrid_rerank"),
                top_k=settings.top_k, threshold=settings.similarity_threshold,
            )
            coverage, reciprocal_rank = evidence_coverage(results, question)
            outputs[mode].append({
                "id": question["id"], "type": question["type"],
                "evidence_recall": coverage, "evidence_mrr": reciprocal_rank,
                "latency_ms": int((time.perf_counter() - tic) * 1000) + (0 if mode == "bm25" else embed_ms),
            })
        if args.generate:
            answer, sources, usage, elapsed = service.answer(
                question["question"], None, f"eval-{question['id']}", []
            )
            terms = question["expected_terms"]
            if question["type"] == "unanswerable":
                correct = ABSTAIN_TEXT in answer
            else:
                correct = all("".join(term.lower().split()) in "".join(answer.lower().split()) for term in terms)
            citation_ids = [int(x) for x in __import__("re").findall(r"\[(\d+)\]", answer)]
            citation_valid = bool(citation_ids) and all(1 <= n <= len(sources) for n in citation_ids)
            generation_results.append({
                "id": question["id"], "type": question["type"], "answer": answer,
                "answer_correct_by_terms": correct,
                "abstention_correct": (ABSTAIN_TEXT in answer) if question["type"] == "unanswerable" else None,
                "citation_valid": citation_valid if question["type"] != "unanswerable" else None,
                "latency_ms": elapsed, "usage": usage,
            })
        print(f"[{idx:02d}/{len(questions)}] {question['id']} complete", flush=True)

    def average(records, field):
        values = [r[field] for r in records if r[field] is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        mode: {
            "evidence_recall_at_k": average(rows, "evidence_recall"),
            "evidence_mrr": average(rows, "evidence_mrr"),
            "mean_latency_ms": average(rows, "latency_ms"),
            "answerable_questions": sum(r["evidence_recall"] is not None for r in rows),
        }
        for mode, rows in outputs.items()
    }
    result = {"summary": summary, "retrieval_details": outputs}
    if args.generate:
        answerable = [r for r in generation_results if r["type"] != "unanswerable"]
        unanswerable = [r for r in generation_results if r["type"] == "unanswerable"]
        result["generation_summary"] = {
            "answer_term_accuracy": average(answerable, "answer_correct_by_terms"),
            "citation_valid_rate": average(answerable, "citation_valid"),
            "abstention_accuracy": average(unanswerable, "abstention_correct"),
            "details": generation_results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

