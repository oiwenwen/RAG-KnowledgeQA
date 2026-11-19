"""Replay fixed conversation prefixes against both memory policies."""
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compact(value):
    return re.sub(r"\s+", "", value)


def rewrite_contains(query, term):
    def normalize(value):
        return re.sub(r"(?:两|二|２)(?=晚)", "2", compact(value))

    return normalize(term) in normalize(query)


def score(case, answer, sources, trace, abstention):
    expected = case.get("action", "search")
    if expected == "clarify":
        candidates = re.split("、", answer.partition("？")[2].rstrip("。"))
        groups = case.get("clarify_candidates", [])
        relevant = not groups or (
            all(any(term in option for term in group for option in candidates) for group in groups)
            and all(any(term in option for group in groups for term in group) for option in candidates)
        )
        return trace["action"] == "clarify" and relevant, None
    if expected == "abstain" and answer == abstention:
        return True, None
    cited = {int(index) for index in re.findall(r"\[(\d+)\]", answer)}
    coverage = all(
        any(i in cited and source["filename"] == filename
            and compact(term) in compact(source["content"])
            for i, source in enumerate(sources, 1))
        for filename, terms in case.get("evidence", {}).items() for term in terms
    )
    if expected == "abstain":
        no_quote = re.search(r"(?:未|不|没有)[^。！？\n]{0,14}(?:报价|价格)|暂无足够依据", answer)
        numeric_price = re.search(r"(?:[0-9０-９]+|[零一二两三四五六七八九十百千万亿]+)\s*(?:元|块钱|美元)|[¥￥$]\s*[0-9０-９]", answer)
        return bool(no_quote and not numeric_price), coverage
    answer_text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", answer)
    return all(compact(term) in compact(answer_text) for term in case["answer"]), coverage


def run(output, case_path=ROOT / "evaluation/dialogue_cases.json", compare_rewrite=False):
    with TemporaryDirectory(prefix="rag-dialogue-") as directory:
        os.environ["RAG_DATA_DIR"] = directory
        from rag_system.config import settings
        from rag_system.service import ABSTAIN_TEXT, KnowledgeService

        service = KnowledgeService()
        service.db.ensure_initial_admin()
        owner = service.db.users()[0]["id"]
        service.seed_demo_documents(owner)
        docs = service.db.documents()
        if len(docs) != 3:
            raise RuntimeError("示例资料未全部入库，请检查密钥和网络。")
        handbook = next(doc for doc in docs if doc["filename"] == "employee_handbook.md")
        fixture = Path(directory) / handbook["filename"]
        original = (ROOT / "demo_data" / fixture.name).read_text(encoding="utf-8")
        fixture.write_text(original.replace("450 元", "500 元"), encoding="utf-8")
        service.replace_document(handbook["id"], fixture, owner)
        cases = json.loads(case_path.read_text(encoding="utf-8"))
        report = {"generation_model": settings.generation_model, "embedding_model": settings.embedding_model,
                  "threshold": settings.similarity_threshold, "top_k": settings.top_k, "runs": {}}
        modes = [("plus", False, "qwen3.7-plus"), ("flash", False, "qwen3.8-flash")] if compare_rewrite else [
            ("recent_four", True, None), ("current", False, None)]
        if compare_rewrite:
            cases += json.loads((ROOT / "evaluation/long_dialogue_cases.json").read_text(encoding="utf-8"))
            cases += json.loads((ROOT / "evaluation/heldout_dialogue_cases.json").read_text(encoding="utf-8"))
        for mode, recent_only, rewrite_model in modes:
            rows = []
            for case in cases:
                questions = case["history"] + case.get("distractors", [])
                history = [{"role": "user", "content": question} for question in questions]
                trace = {}
                answer, sources, _, elapsed = service.answer(
                    case["question"], None, case["id"], history, recent_only=recent_only, trace=trace,
                    rewrite_model=rewrite_model,
                )
                matched, coverage = score(case, answer, sources, trace, ABSTAIN_TEXT)
                rewrite_check = all(rewrite_contains(trace["retrieval_query"], term)
                                    for term in case["rewrite_terms"]) if "rewrite_terms" in case else None
                rows.append(dict(id=case["id"], answer=answer, answer_check=matched, rewrite_check=rewrite_check,
                                 citation_evidence_check=coverage, latency_ms=elapsed, **trace))
                print(mode, case["id"], matched, coverage, flush=True)
            report["runs"][mode] = dict(
                rewrite_model=rewrite_model or settings.rewrite_model,
                passed=sum(not row.get("rewrite_request_failed") and row["answer_check"] and row["citation_evidence_check"] is not False
                           and row["rewrite_check"] is not False for row in rows),
                total=len(rows), mean_latency_ms=sum(row["latency_ms"] for row in rows) / len(rows), cases=rows,
            )
            stages = {}
            for row in rows:
                for stage, elapsed in row["timings_ms"].items():
                    stages.setdefault(stage, []).append(elapsed)
            report["runs"][mode]["stage_latency_ms"] = {
                stage: {"calls": len(times), "mean_when_run": round(sum(times) / len(times), 2),
                        "mean_per_case": round(sum(times) / len(rows), 2)}
                for stage, times in stages.items()
            }
            usage_rows = [row["rewrite_usage"] for row in rows if "rewrite_usage" in row]
            report["runs"][mode]["rewrite_tokens"] = {
                "recorded_calls": len(usage_rows),
                **{key: sum(row.get(key, 0) for row in usage_rows)
                   for key in ("input_tokens", "output_tokens", "total_tokens")},
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/dialogue-results.json",
        Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "evaluation/dialogue_cases.json",
        compare_rewrite="--compare-rewrite" in sys.argv[3:])
