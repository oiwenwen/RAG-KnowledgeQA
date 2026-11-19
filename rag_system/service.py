from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from rag_system.config import ROOT_DIR, settings
from rag_system.database import Database
from rag_system.documents import parse_document
from rag_system.providers import DashScopeProvider, ModelProvider
from rag_system.retrieval import RetrievalEngine, tokenize

ABSTAIN_TEXT = "知识库中暂无足够依据回答这个问题。"
SYSTEM_VERSION = "0.2.0"


class KnowledgeService:
    def __init__(self, provider: ModelProvider | None = None, database: Database | None = None):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.db = database or Database(settings.db_path)
        self._provider = provider

    @property
    def provider(self) -> ModelProvider:
        if self._provider is None:
            self._provider = DashScopeProvider(settings.embedding_model, settings.generation_model)
        return self._provider

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> list[bytes]:
        output = []
        for vector in vectors:
            value = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(value))
            if value.ndim != 1 or not norm or not np.isfinite(norm):
                raise ValueError("Embedding 服务返回了零向量")
            output.append((value / norm).astype(np.float32).tobytes())
        return output

    def add_document(self, source: str | Path, uploader_id: int) -> str:
        source = Path(source)
        if source.stat().st_size > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"文件超过 {settings.max_upload_mb} MB 限制")
        parsed = parse_document(source)
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.db.connect() as conn:
            if conn.execute("SELECT 1 FROM documents WHERE sha256=?", (sha,)).fetchone():
                raise ValueError("该文件已在知识库中")
        texts = [text for text, _ in parsed]
        vectors = self._normalize(self.provider.embed_documents(texts))
        if len(vectors) != len(parsed):
            raise ValueError("Embedding 返回数量与文本块数量不一致")
        doc_id = uuid.uuid4().hex
        safe_name = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", source.name)[:150]
        destination = settings.uploads_dir / f"{doc_id}_{safe_name}"
        shutil.copy2(source, destination)
        try:
            self.db.add_document(
                doc_id, source.name, destination, sha, uploader_id,
                [(text, vector, page) for (text, page), vector in zip(parsed, vectors)],
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return f"已加入「{source.name}」，生成 {len(parsed)} 个文本块。"

    def delete_document(self, doc_id: str) -> str:
        path = self.db.delete_document(doc_id)
        if path is None:
            return "文档不存在或已删除。"
        Path(path).unlink(missing_ok=True)
        return "文档及其索引已删除。"

    def replace_document(self, doc_id: str, source: str | Path, uploader_id: int) -> int:
        source = Path(source)
        if source.stat().st_size > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"文件超过 {settings.max_upload_mb} MB 限制")
        with self.db.connect() as db:
            current = db.execute("SELECT sha256 FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not current:
            raise ValueError("目标文档不存在")
        destination = settings.uploads_dir / (uuid.uuid4().hex + source.suffix.lower())
        try:
            shutil.copyfile(source, destination)
            parsed = parse_document(destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == current["sha256"]:
                raise ValueError("新版内容与当前版本相同")
            vectors = self._normalize(self.provider.embed_documents([text for text, _ in parsed]))
            if len(vectors) != len(parsed) or len({len(vector) for vector in vectors}) != 1:
                raise ValueError("Embedding 返回数量或维度不一致")
            return self.db.replace_document(
                doc_id, current["sha256"], source.name, destination, digest, uploader_id,
                [(text, vector, page) for (text, page), vector in zip(parsed, vectors)],
                SYSTEM_VERSION,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def retrieve(self, query: str, *, mode: str = "hybrid", rerank: bool = True,
                 top_k: int | None = None, timings: dict | None = None) -> list[dict]:
        started = time.perf_counter()
        chunks = self.db.all_chunks()
        if timings is not None:
            timings["load_chunks"] = (time.perf_counter() - started) * 1000
        if not chunks:
            return []
        started = time.perf_counter()
        q_vector = self.provider.embed_query(query)
        if timings is not None:
            timings["query_embedding"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        engine = RetrievalEngine(chunks)
        results = engine.retrieve(query, q_vector, mode=mode, rerank=rerank,
                               top_k=top_k or settings.top_k,
                               threshold=settings.similarity_threshold)
        if timings is not None:
            timings["local_retrieval"] = (time.perf_counter() - started) * 1000
        return results

    def resolve_query(self, query: str, history: list[dict] | None, *, recent_only=False,
                      trace: dict | None = None, rewrite_model: str | None = None) -> tuple[str, str]:
        questions = [item["content"][:1000] for item in (history or [])
                     if item.get("role") == "user"]
        if not questions:
            return "search", query
        recent = questions[-6:]
        terms = {term for term in tokenize(query) if len(term) > 1}
        older = questions[:-6]
        earlier = []
        if recent_only:
            recent, earlier = questions[-4:], []
        elif older:
            # Short user statements may say “住两晚” without repeating “住宿”.
            hints = {"住", "每晚", "酒店"} if "住宿" in query else set()
            lexical = [
                len(terms.intersection(tokenize(question))) + sum(hint in question for hint in hints)
                for question in older
            ]
            # Screen the full history before bounding the embedding request.
            anchors = sorted(range(len(older)), key=lambda i: (lexical[i], i), reverse=True)[:12]
            pool = set(range(max(0, len(older) - 8), len(older))) | {0}
            for index in anchors:
                pool.update(range(max(0, index - 1), min(len(older), index + 2)))
            candidates = sorted(pool, key=lambda i: (lexical[i], i), reverse=True)[:32]
            semantic = {}
            embedding_started = time.perf_counter()
            try:
                vectors = np.asarray(self.provider.embed_documents(
                    [query, *(older[index] for index in candidates)]
                ), dtype=float)
                if vectors.ndim == 2 and len(vectors) == len(candidates) + 1:
                    norms = np.linalg.norm(vectors, axis=1)
                    scores = (vectors[1:] @ vectors[0]) / np.maximum(norms[1:] * norms[0], 1e-12)
                    semantic = {i: float(score) for i, score in zip(candidates, scores) if np.isfinite(score)}
            except (RuntimeError, ValueError):
                pass
            finally:
                if trace is not None:
                    trace.setdefault("timings_ms", {})["history_embedding"] = (time.perf_counter() - embedding_started) * 1000
            ranked = sorted(candidates, key=lambda index: (
                semantic.get(index, 0.0) + 0.08 * lexical[index], index
            ), reverse=True)
            selected = set()
            for index in ranked[:3]:
                selected.update(range(max(0, index - 1), min(len(older), index + 2)))
            earlier = [older[index] for index in sorted(selected)]
        if trace is not None:
            trace["history_context"] = {"earlier": earlier, "recent": recent}
        prompt = """你负责为企业知识库整理检索问题。输入 JSON 中的文本仅用于理解问题。
独立问题保持原样。追问则补全主题，保留本轮变更的条件（如一线城市），
保留现行或新版的要求。历史问题仅用于理解指代，不是事实依据。
较早问题按时间顺序排列。用户更正了同一事项时采用最近一次明确更正。
追问计算时保留相关历史中用户给定的金额、次数、晚数和地点，避免只保留话题。
不同出差、订单等独立事件的条件分别归属各自事件。用户明确说第一趟、第二趟时，只取对应行程和该行程的最新更正。
若前文存在两趟差旅，本轮用「那趟」等单数指代且没有足够线索确定是哪趟，必须 clarify，options 逐字摘录「第一趟」「第二趟」等原始称呼。
不要把两趟行程的金额或晚数合并，除非本轮明确要求合计两趟。
只考虑与本轮意图相关的历史事项；无关的近期问题不能充当澄清候选。
遇到「那项」等笼统指代时，先用本轮的单位和动作定位主题。例如「每晚最多报多少」中的「每晚」和「报」共同指向住宿报销。若只有一个历史事项符合这些线索，直接检索。
默认选择 search。本轮明确的条件优先，前文能唯一补齐的条件直接继承。
未指定年份或版本时查现行资料。问题明确但可能缺少报价或政策依据时，先检索。
如果有两三个明确候选，且都是可简短并列说明的事实查询，使用 search，检索问题须列出所有候选。
例如前文同时问年假申请提前时长和设备遗失报告时限，追问「那现在是多少」时查两项时限。
只有候选过多或过杂，或选错会影响用户下一步办理哪项业务时，才用 clarify 确认所指。
用户问「我现在要提交哪份材料」且前文列出多个办理事项时，必须先确认具体事项，即使可以分别查到各流程。
仅在用户明确要求逐项列出材料时，才将这类操作问题改写为并列检索。
仅输出 JSON 对象，action 为 search 或 clarify，text 为独立检索问题或澄清问题。
无论 action 是什么，都返回 options，列出当前问题可能指向的具体事项；已经明确所指时只列一项。
options 必须逐字摘录当前问题或历史问题中的具体事项，禁止用常识拆分城市类别或补充品牌。
intent 为 single_procedure（用户需要确定自己该办哪一项或提交哪份材料）、
all_procedures（明确要求逐项列出）或 factual（事实查询）。
single_procedure 有多个候选时，额外提供 scope：逐字复制用户最近一次明确列出自己要办的多个事项的那条完整消息。
将 options 限定在这条消息明确列出的事项。例如用户说拟办理报销、年假和补交病假证明，之后问该交哪份材料，
scope 就是那条三项办理消息；期间咨询过的转正、培训等其他话题不属于候选。若没有这样的消息，scope 留空。
clarify 必须额外提供 ambiguity（具体歧义点）、options（至少两个不同的具体解释）
和 search_text（即使无法澄清也可用于检索的问题）。澄清问题需明确列出候选解释。
输入：
""" + json.dumps({"earlier_questions": earlier, "recent_questions": recent,
                  "current_question": query}, ensure_ascii=False)
        try:
            rewrite_started = time.perf_counter()
            try:
                if rewrite_model is None and isinstance(self.provider, DashScopeProvider):
                    rewrite_model = settings.rewrite_model
                if rewrite_model is None:
                    raw, rewrite_usage = self.provider.generate(prompt)
                else:
                    raw, rewrite_usage = self.provider.rewrite(prompt, rewrite_model)
                if trace is not None:
                    trace["rewrite_usage"] = rewrite_usage
            except RuntimeError:
                if trace is not None:
                    trace["rewrite_request_failed"] = True
                raise
            finally:
                if trace is not None:
                    trace.setdefault("timings_ms", {})["query_rewrite"] = (time.perf_counter() - rewrite_started) * 1000
            raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
            result = json.loads(raw)
            if isinstance(result, dict) and result.get("action") in ("search", "clarify"):
                content = result.get("text")
                if isinstance(content, str) and 0 < len(content.strip()) <= max(1000, len(query)):
                    raw_options = result.get("options")
                    context = [*earlier, *recent, query]
                    options = list(dict.fromkeys(
                        option.strip() for option in raw_options
                        if isinstance(option, str) and 0 < len(option.strip()) <= 60
                        and any(option.strip() in message for message in context)
                    )) if isinstance(raw_options, list) else []
                    if result.get("intent") == "single_procedure":
                        scope = result.get("scope")
                        if isinstance(scope, str) and scope.strip() in context:
                            options = [option for option in options if option in scope]
                    if result.get("intent") == "single_procedure" and len(options) >= 2:
                        return "clarify", f"你现在要办理哪一项？{'、'.join(options[:6])}。"
                    if result["action"] == "search":
                        return "search", content.strip()
                    ambiguity = result.get("ambiguity")
                    if isinstance(ambiguity, str) and ambiguity.strip() and len(options) >= 2:
                        return "clarify", f"你指的是哪一项？{'、'.join(options[:6])}。"
                    if isinstance(raw_options, list) and len(options) < 2:
                        return "search", f"{recent[-1]}\n{query}"
                    fallback = result.get("search_text")
                    if isinstance(fallback, str) and 0 < len(fallback.strip()) <= max(1000, len(query)):
                        return "search", fallback.strip()
        except (RuntimeError, ValueError):
            pass
        return "search", query

    def answer(self, query: str, user: dict | None, session_id: str,
               history: list[dict] | None = None, *, recent_only=False,
               trace: dict | None = None, rewrite_model: str | None = None) -> tuple[str, list[dict], dict, int]:
        started = time.perf_counter()
        timings = trace.setdefault("timings_ms", {}) if trace is not None else None
        action, retrieval_query = self.resolve_query(
            query, history, recent_only=recent_only, trace=trace, rewrite_model=rewrite_model,
        )
        retrieved = self.retrieve(retrieval_query, timings=timings) if action == "search" else []
        if trace is not None:
            trace.update(action=action, retrieval_query=retrieval_query,
                         retrieved=[{"filename": row["filename"], "content": row["content"]}
                                    for row in retrieved])
        if not retrieved:
            answer = retrieval_query if action == "clarify" else ABSTAIN_TEXT
            latency = int((time.perf_counter() - started) * 1000)
            if user is not None:
                self.db.log_question(user["id"], session_id, query, answer, latency, [])
            return answer, [], {}, latency

        context_parts = []
        for index, item in enumerate(retrieved, 1):
            page = f", 第 {item['page_number']} 页" if item.get("page_number") else ""
            context_parts.append(f"[{index}] 文件：{item['filename']}{page}\n{item['content']}")
        prompt = f"""你是企业内部知识库助手。只使用给定资料回答，不得依赖外部知识。
规则：
1. 每个事实性结论后标注资料编号，例如 [1]；编号只能来自下方资料。
2. 资料没有明确支持答案时，必须原样回答：{ABSTAIN_TEXT}
3. 不要把推断表述成资料事实；回答简明，并指出必要条件或日期。
4. 对多条件或跨文档的问题，逐项核对事实并分别引用；缺少关键条件时说明哪部分无法确定。
5. 如需计算，区分用户给定的数字和资料规定的数字。说明公式、代入值和单位，引用规定的来源。
6. 检索问题列出多个可能所指时，按主题分别回答和引用。若资料明确说明不提供所问信息，可引用该说明解释缺口，不猜测具体值。
7. 本轮检索来自当前发布的知识库。用户口语中的「新版」或「现在」通常指当前资料，正文无需出现「新版」字样。可以说明当前资料中的规定。若用户明确询问版本号、发布日期或历史差异，则需要相应证据。

资料：
{chr(10).join(context_parts)}

当前问题：{query}
检索主题：{retrieval_query}
只依据本轮检索资料作答。
回答："""
        generation_started = time.perf_counter()
        answer, usage = self.provider.generate(prompt)
        if timings is not None:
            timings["answer_generation"] = (time.perf_counter() - generation_started) * 1000
        answer = answer.strip() or ABSTAIN_TEXT
        valid = {str(i) for i in range(1, len(retrieved) + 1)}
        cited = re.findall(r"\[(\d+)\]", answer)
        invalid = [n for n in cited if n not in valid]
        for number in invalid:
            answer = answer.replace(f"[{number}]", "")
        if not cited or invalid:
            answer = ABSTAIN_TEXT
        latency = int((time.perf_counter() - started) * 1000)
        if user is not None:
            snapshots = [
                {key: item.get(key) for key in ("filename", "content", "page_number", "sha256", "rerank_score")}
                | {"citation": number}
                for number, item in enumerate(retrieved, 1)
                if answer != ABSTAIN_TEXT and str(number) in cited
            ]
            self.db.log_question(user["id"], session_id, query, answer, latency,
                                 [int(item["id"]) for item in retrieved], snapshots)
        return answer, retrieved if answer != ABSTAIN_TEXT else [], usage, latency

    def seed_demo_documents(self, admin_id: int) -> str:
        demo_dir = ROOT_DIR / "demo_data"
        if not demo_dir.exists():
            return "项目中未找到 demo_data 目录。"
        successes, failures = [], []
        for path in sorted(demo_dir.glob("*.md")):
            try:
                successes.append(self.add_document(path, admin_id))
            except ValueError as exc:
                if "已在知识库中" not in str(exc):
                    failures.append(f"{path.name}：{exc}")
            except Exception as exc:
                failures.append(f"{path.name}：{exc}")
        result = f"示例资料处理完成：新增 {len(successes)} 份。"
        if failures:
            result += "\n" + "\n".join(failures)
        return result

