"""Copy a stopped application's knowledge base into an independent directory."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory

from rag_system.config import settings

DATABASE = "knowledge.db"
MANIFEST = "backup.json"
DOCUMENTS = "documents"
FORMAT_VERSION = 1


def file_digest(path: Path) -> str:
    with path.open("rb") as uploaded:
        return hashlib.file_digest(uploaded, "sha256").hexdigest()


def copy_database(source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(destination)) as copy:
            original.backup(copy)


def check_database(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("备份数据库未通过完整性校验")
    if connection.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("备份数据库存在失效的数据引用")


def copy_documents(connection, source, stage, destination, *, restoring):
    copied = {}
    documents_root = source / DOCUMENTS
    output = stage / DOCUMENTS
    output.mkdir()

    def move_reference(stored_path, digest):
        stored = Path(stored_path)
        relative = Path(DOCUMENTS) / stored.name
        expected = relative if restoring else source / relative
        if stored != expected or not stored.name:
            raise ValueError("文档路径不在当前数据目录中，请确认源库路径正确")
        original = source / relative
        if original.resolve().parent != documents_root or original.is_symlink() or not original.is_file():
            raise ValueError(f"原始文档缺失或路径异常：{stored.name}")
        if stored.name not in copied:
            target = output / stored.name
            shutil.copyfile(original, target)
            copied[stored.name] = file_digest(target)
        if copied[stored.name] != digest:
            raise ValueError(f"原始文档指纹不匹配：{stored.name}")
        return str(destination / relative) if restoring else relative.as_posix()

    for doc_id, stored_path, digest in connection.execute("SELECT id, stored_path, sha256 FROM documents").fetchall():
        path = move_reference(stored_path, digest)
        connection.execute("UPDATE documents SET stored_path=? WHERE id=?", (path, doc_id))
    for release_id, raw in connection.execute("SELECT id, previous_document FROM knowledge_releases").fetchall():
        previous = json.loads(raw)
        previous["stored_path"] = move_reference(previous["stored_path"], previous["sha256"])
        connection.execute("UPDATE knowledge_releases SET previous_document=? WHERE id=?",
                           (json.dumps(previous, ensure_ascii=False), release_id))
    return len(copied)


def transfer(source: Path, destination: Path, *, restoring: bool = False) -> int:
    source = source.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("目标目录已经存在，请选择一个新目录")
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("目标目录不能放在源数据目录中")
    database = source / DATABASE
    if not database.is_file() or database.is_symlink():
        raise ValueError("源目录没有有效的 knowledge.db")
    if restoring:
        manifest = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("无法识别备份格式版本")
        if manifest.get("database_sha256") != file_digest(database):
            raise ValueError("数据库备份指纹不匹配")
        if any((source / (DATABASE + suffix)).exists() for suffix in ("-wal", "-journal")):
            raise ValueError("备份目录中存在数据库日志，请使用未被应用打开过的备份")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rag-backup-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        snapshot = stage / DATABASE
        copy_database(database, snapshot)
        with closing(sqlite3.connect(snapshot)) as connection:
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA journal_mode=DELETE")
            check_database(connection)
            with connection:
                count = copy_documents(connection, source, stage, destination, restoring=restoring)
        if not restoring:
            manifest = {"format_version": FORMAT_VERSION, "database_sha256": file_digest(snapshot)}
            (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        destination.mkdir()
        placed = []
        try:
            for item in stage.iterdir():
                target = destination / item.name
                item.rename(target)
                placed.append((item, target))
        except OSError:
            for item, target in reversed(placed):
                target.rename(item)
            destination.rmdir()
            raise
    return count


def main():
    parser = argparse.ArgumentParser(description="离线备份及恢复知识库；请先关闭应用。")
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup", help="备份当前数据库及原件")
    backup.add_argument("destination", type=Path)
    backup.add_argument("--source", type=Path, default=settings.data_dir)
    restore = commands.add_parser("restore", help="恢复到新数据目录，不覆盖原库")
    restore.add_argument("source", type=Path)
    restore.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        count = transfer(args.source, args.destination, restoring=args.command == "restore")
    except (OSError, ValueError, sqlite3.Error, KeyError, TypeError) as exc:
        parser.exit(1, f"{'恢复' if args.command == 'restore' else '备份'}失败：{exc}\n")
    print(f"已{'恢复' if args.command == 'restore' else '备份'}到 {args.destination.resolve()}，包含 {count} 份原始文档。")


if __name__ == "__main__":
    main()
