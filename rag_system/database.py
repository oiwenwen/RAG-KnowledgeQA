from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Iterator

from rag_system.auth import hash_password, verify_password


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    login_version INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    uploaded_by INTEGER REFERENCES users(id),
                    created_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    page_number INTEGER,
                    UNIQUE(document_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS question_logs (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    retrieved_chunk_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_releases (
                    id INTEGER PRIMARY KEY,
                    system_version TEXT NOT NULL,
                    published_by INTEGER NOT NULL REFERENCES users(id),
                    published_at TEXT NOT NULL,
                    previous_document TEXT NOT NULL,
                    manifest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS answer_sources (
                    log_id INTEGER PRIMARY KEY REFERENCES question_logs(id) ON DELETE CASCADE,
                    snapshot TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_logs_owner_session
                    ON question_logs(user_id, session_id, id);
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_logs_created ON question_logs(created_at);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
            if "login_version" not in columns:
                db.execute("ALTER TABLE users ADD COLUMN login_version INTEGER NOT NULL DEFAULT 0")

    def ensure_initial_admin(self) -> tuple[str, str] | None:
        with self.connect() as db:
            exists = db.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone()
            if exists:
                return None
            username = "admin"
            password = secrets.token_urlsafe(15)
            salt, digest = hash_password(password)
            db.execute("INSERT INTO users(username,password_salt,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                       (username, salt, digest, "admin", utc_now()))
            return username, password

    def authenticate(self, username: str, password: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE username=? AND active=1", (username.strip(),)).fetchone()
        if not row or not verify_password(password, row["password_salt"], row["password_hash"]):
            return None
        return {key: row[key] for key in ("id", "username", "role", "login_version")}

    def user_login_state(self, user_id: int) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute("SELECT role, active, login_version FROM users WHERE id=?", (user_id,)).fetchone()

    def set_user_active(self, user_id: int, active: bool) -> bool:
        with self.connect() as db:
            result = db.execute("""
                UPDATE users SET active=?, login_version=login_version+1
                WHERE id=? AND role='user' AND active<>?
            """, (int(active), user_id, int(active)))
            if result.rowcount:
                return True
            target = db.execute("SELECT 1 FROM users WHERE id=? AND role='user'", (user_id,)).fetchone()
            if not target:
                raise ValueError("请选择有效的普通用户")
            return False

    def reset_user_password(self, user_id: int, password: str) -> None:
        salt, digest = hash_password(password)
        with self.connect() as db:
            result = db.execute("""
                UPDATE users SET password_salt=?, password_hash=?, login_version=login_version+1
                WHERE id=? AND role='user'
            """, (salt, digest, user_id))
            if not result.rowcount:
                raise ValueError("请选择有效的普通用户")

    def create_user(self, username: str, password: str, role: str) -> None:
        username = username.strip()
        if not username or len(username) > 64:
            raise ValueError("用户名不能为空且不能超过 64 个字符")
        if role not in {"admin", "user"}:
            raise ValueError("角色无效")
        salt, digest = hash_password(password)
        with self.connect() as db:
            db.execute("INSERT INTO users(username,password_salt,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                       (username, salt, digest, role, utc_now()))

    def users(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT id,username,role,active,created_at FROM users ORDER BY id")]

    def add_document(self, doc_id: str, filename: str, path: Path, sha256: str,
                     uploader_id: int, chunks: list[tuple[str, bytes, int | None]]) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO documents(id,filename,stored_path,sha256,uploaded_by,created_at,chunk_count) VALUES(?,?,?,?,?,?,?)",
                       (doc_id, filename, str(path), sha256, uploader_id, utc_now(), len(chunks)))
            db.executemany("INSERT INTO chunks(document_id,ordinal,content,vector,page_number) VALUES(?,?,?,?,?)",
                           [(doc_id, i, content, vector, page) for i, (content, vector, page) in enumerate(chunks)])

    def documents(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT d.id,d.filename,d.sha256,d.created_at,d.chunk_count,u.username AS uploaded_by
                FROM documents d LEFT JOIN users u ON u.id=d.uploaded_by ORDER BY d.created_at DESC
            """)]

    def replace_document(self, doc_id, expected_hash, filename, path, digest,
                         uploader_id, chunks, system_version):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not old or old["sha256"] != expected_hash:
                raise ValueError("目标文档已变化，请刷新列表后重试")
            if digest == expected_hash:
                raise ValueError("新版内容与当前版本相同")
            if db.execute("SELECT 1 FROM documents WHERE sha256=?", (digest,)).fetchone():
                raise ValueError("新版内容已作为其他文档入库")
            dimension = db.execute("SELECT length(vector) FROM chunks WHERE document_id != ? LIMIT 1",
                                   (doc_id,)).fetchone()
            if dimension and any(len(vector) != dimension[0] for _, vector, _ in chunks):
                raise ValueError("新向量与知识库维度不同")
            now = utc_now()
            db.execute("UPDATE documents SET filename=?, stored_path=?, sha256=?, uploaded_by=?, created_at=?, chunk_count=? WHERE id=?",
                       (filename, str(path), digest, uploader_id, now, len(chunks), doc_id))
            db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            db.executemany("INSERT INTO chunks(document_id,ordinal,content,vector,page_number) VALUES(?,?,?,?,?)",
                           [(doc_id, i, text, vector, page) for i, (text, vector, page) in enumerate(chunks)])
            manifest = [dict(row) for row in db.execute(
                "SELECT id,filename,sha256,chunk_count FROM documents ORDER BY id")]
            cursor = db.execute(
                "INSERT INTO knowledge_releases(system_version,published_by,published_at,previous_document,manifest) VALUES(?,?,?,?,?)",
                (system_version, uploader_id, now, json.dumps(dict(old), ensure_ascii=False),
                 json.dumps(manifest, ensure_ascii=False)))
            return cursor.lastrowid

    def knowledge_releases(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT r.*, u.username FROM knowledge_releases r
                JOIN users u ON u.id = r.published_by
                ORDER BY r.id DESC LIMIT ?
            """, (limit,))]

    def delete_document(self, doc_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT stored_path FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            return row["stored_path"]

    def all_chunks(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT c.id,c.document_id,c.ordinal,c.content,c.vector,c.page_number,d.filename,d.sha256
                FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.id
            """)]

    def log_question(self, user_id: int, session_id: str, question: str, answer: str,
                     latency_ms: int, chunk_ids: list[int], sources: list[dict] | None = None) -> None:
        with self.connect() as db:
            cursor = db.execute("INSERT INTO question_logs(user_id,session_id,question,answer,latency_ms,retrieved_chunk_ids,created_at) VALUES(?,?,?,?,?,?,?)",
                       (user_id, session_id, question, answer, latency_ms,
                        ",".join(map(str, chunk_ids)), utc_now()))
            db.execute("INSERT INTO answer_sources VALUES (?,?)",
                       (cursor.lastrowid, json.dumps(sources or [], ensure_ascii=False)))

    def sessions(self, user_id: int):
        with self.connect() as db:
            return db.execute("""
                SELECT q.session_id, q.question FROM question_logs q
                JOIN (SELECT MIN(id) AS first_id, MAX(id) AS last_id
                      FROM question_logs WHERE user_id=? GROUP BY session_id) s
                ON q.id=s.first_id ORDER BY s.last_id DESC
            """, (user_id,)).fetchall()

    def conversation(self, user_id: int, session_id: str):
        with self.connect() as db:
            return db.execute("""
                SELECT q.question,q.answer,q.latency_ms,s.snapshot FROM question_logs q
                LEFT JOIN answer_sources s ON q.id=s.log_id
                WHERE q.user_id=? AND q.session_id=? ORDER BY q.id
            """, (user_id, session_id)).fetchall()

    def recent_logs(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT l.created_at,u.username,l.question,l.answer,l.latency_ms
                FROM question_logs l LEFT JOIN users u ON u.id=l.user_id
                ORDER BY l.id DESC LIMIT ?
            """, (limit,))]

