"""Offline acceptance check for publication and durable citation snapshots."""
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FixtureProvider:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]

    def generate(self, prompt):
        limit = re.search(r"其他城市每晚不超过 (\d+) 元", prompt).group(1)
        return f"其他城市每晚不超过 {limit} 元 [1]。", {}


def run(directory):
    os.environ["RAG_DATA_DIR"] = directory
    from rag_system.config import settings
    from rag_system.service import KnowledgeService

    provider = FixtureProvider()
    service = KnowledgeService(provider)
    service.db.ensure_initial_admin()
    admin = next(user for user in service.db.users() if user["role"] == "admin")
    service.db.create_user("acceptance", "acceptance-demo-password", "user")
    reader = service.db.authenticate("acceptance", "acceptance-demo-password")
    original = ROOT / "demo_data/employee_handbook.md"
    service.add_document(original, admin["id"])
    doc_id = service.db.documents()[0]["id"]
    question = "其他城市每晚住宿报销上限是多少？"

    def ask(session, expected):
        answer, sources, _, _ = service.answer(question, reader, session)
        assert f"{expected} 元 [1]" in answer and f"{expected} 元" in sources[0]["content"]

    def snapshot(session):
        row, = service.db.conversation(reader["id"], session)
        return row["answer"], json.loads(row["snapshot"])

    ask("before", 450)
    old_snapshot = snapshot("before")
    updated = Path(directory) / original.name
    updated.write_text(original.read_text(encoding="utf-8").replace("450 元", "500 元"), encoding="utf-8")
    service.replace_document(doc_id, updated, admin["id"])
    service = KnowledgeService(provider)
    ask("after", 500)
    assert snapshot("before") == old_snapshot
    assert old_snapshot[1][0]["sha256"] != snapshot("after")[1][0]["sha256"]
    release, = service.db.knowledge_releases()
    previous = json.loads(release["previous_document"])
    published, = json.loads(release["manifest"])
    assert release["username"] == admin["username"]
    assert previous["id"] == published["id"] == doc_id
    assert previous["sha256"] == old_snapshot[1][0]["sha256"]
    assert published["sha256"] == snapshot("after")[1][0]["sha256"]
    assert service.db.documents()[0]["sha256"] == published["sha256"]
    print("PASS: 450 → 500 after reopening SQLite; historical citation unchanged")

    def state():
        with service.db.connect() as connection:
            releases = connection.execute("SELECT COUNT(*) FROM knowledge_releases").fetchone()[0]
        return service.db.documents(), service.db.all_chunks(), releases, sorted(settings.uploads_dir.iterdir())

    expected_state = state()
    empty = Path(directory) / "empty.md"
    empty.touch()
    updated.write_text(updated.read_text(encoding="utf-8").replace("500 元", "550 元"), encoding="utf-8")
    for label, candidate, error, message in (
        ("empty document", empty, ValueError, "没有可索引的文本"),
        ("embedding failure", updated, RuntimeError, "simulated embedding timeout"),
    ):
        if error is RuntimeError:
            with patch.object(provider, "embed_documents", side_effect=error(message)):
                expect_failure(service, doc_id, candidate, admin["id"], error, message)
        else:
            expect_failure(service, doc_id, candidate, admin["id"], error, message)
        assert state() == expected_state
        ask(label, 500)
        assert snapshot("before") == old_snapshot
        print(f"PASS: {label}; active index, release log and uploaded files unchanged")


def expect_failure(service, doc_id, candidate, uploader, error, message):
    try:
        service.replace_document(doc_id, candidate, uploader)
    except error as exc:
        assert message in str(exc)
    else:
        raise AssertionError("Replacement unexpectedly succeeded")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="rag-update-check-") as directory:
        run(directory)
