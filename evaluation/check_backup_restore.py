"""Offline backup and recovery acceptance using disposable knowledge bases."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from check_knowledge_update import FixtureProvider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def expect_failure(error, message, source, destination, *, restoring=False):
    from rag_system.backup import transfer

    try:
        transfer(source, destination, restoring=restoring)
    except error as exc:
        assert message in str(exc)
    else:
        raise AssertionError("Unexpected successful transfer")


def run(root):
    active = root / "active"
    os.environ["RAG_DATA_DIR"] = str(active)
    from rag_system.backup import file_digest, transfer
    from rag_system.config import settings
    from rag_system.service import KnowledgeService

    provider = FixtureProvider()
    service = KnowledgeService(provider)
    admin_name, admin_password = service.db.ensure_initial_admin()
    admin = service.db.authenticate(admin_name, admin_password)
    service.db.create_user("backup-reader", "fixture-reader-password", "user")
    reader = service.db.authenticate("backup-reader", "fixture-reader-password")
    original = ROOT / "demo_data/employee_handbook.md"
    question = "其他城市每晚住宿报销上限是多少？"
    service.add_document(original, admin["id"])
    old_answer = service.answer(question, reader, "old")[0]
    assert "450 元 [1]" in old_answer
    revision = root / original.name
    revision.write_text(original.read_text(encoding="utf-8").replace("450 元", "500 元"), encoding="utf-8")
    doc_id = service.db.documents()[0]["id"]
    service.replace_document(doc_id, revision, admin["id"])
    service.answer(question, reader, "new")
    old_turn, = service.db.conversation(reader["id"], "old")
    new_turn, = service.db.conversation(reader["id"], "new")
    expected_chunks = service.db.all_chunks()
    release, = service.db.knowledge_releases()

    (active / ".env").write_text("FIXTURE_ONLY=true\n", encoding="utf-8")
    (active / "question_logs_export.jsonl").write_text("fixture\n", encoding="utf-8")
    backup = root / "backup"
    assert transfer(active, backup) == 2
    assert {path.name for path in backup.iterdir()} == {"backup.json", "knowledge.db", "documents"}
    assert dict(service.db.conversation(reader["id"], "old")[0]) == dict(old_turn)
    moved_backup = backup.rename(root / "moved backup")
    moved_active = active.rename(root / "offline original")
    print("PASS: backup excludes local config and exports; archive is portable")

    restored = root / "restored"
    assert transfer(moved_backup, restored, restoring=True) == 2
    with patch("rag_system.service.settings", replace(settings, data_dir=restored)):
        recovered = KnowledgeService(provider)
        recovered_reader = recovered.db.authenticate("backup-reader", "fixture-reader-password")
        assert recovered_reader == reader
        assert recovered.db.authenticate(admin_name, admin_password) == admin
        assert {row["session_id"] for row in recovered.db.sessions(reader["id"])} == {"old", "new"}
        assert dict(recovered.db.conversation(reader["id"], "old")[0]) == dict(old_turn)
        assert dict(recovered.db.conversation(reader["id"], "new")[0]) == dict(new_turn)
        assert recovered.db.all_chunks() == expected_chunks
        assert "500 元 [1]" in recovered.answer(question, recovered_reader, "restored")[0]
        restored_release, = recovered.db.knowledge_releases()
        assert restored_release["manifest"] == release["manifest"]
        previous = json.loads(restored_release["previous_document"])
        previous_path = Path(previous["stored_path"])
        assert previous_path.parent == restored / "documents"
        assert file_digest(previous_path) == previous["sha256"]
        with recovered.db.connect() as connection:
            document = connection.execute("SELECT stored_path, sha256 FROM documents").fetchone()
        active_path = Path(document["stored_path"])
        assert active_path.parent == restored / "documents"
        assert file_digest(active_path) == document["sha256"]
        recovered.delete_document(doc_id)
        assert not active_path.exists()
        assert file_digest(moved_active / "documents" / active_path.name) == document["sha256"]
    print("PASS: independent login, search, citations, publication and document paths")

    expect_failure(FileExistsError, "目标目录已经存在", moved_backup, restored, restoring=True)
    assert (restored / "knowledge.db").is_file()
    backup_file = moved_backup / "documents" / active_path.name
    content = backup_file.read_bytes()
    rejected = root / "rejected"
    backup_file.write_bytes(content + b"\ncorrupted")
    expect_failure(ValueError, "原始文档指纹不匹配", moved_backup, rejected, restoring=True)
    assert not rejected.exists()
    backup_file.unlink()
    expect_failure(ValueError, "原始文档缺失", moved_backup, rejected, restoring=True)
    assert not rejected.exists()
    backup_file.write_bytes(content)
    with (moved_backup / "knowledge.db").open("ab") as archive:
        archive.write(b"corrupted")
    expect_failure(ValueError, "数据库备份指纹不匹配", moved_backup, rejected, restoring=True)
    assert not rejected.exists()
    print("PASS: no overwrite; missing files and corrupted backups are rejected")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="rag-backup-check-") as directory:
        run(Path(directory))
