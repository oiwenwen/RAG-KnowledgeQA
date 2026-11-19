"""Exercise account callbacks against an isolated, pre-upgrade SQLite database."""
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_system.auth import hash_password


def setup_legacy_users(database, admin_password, reader_password):
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE users (
            id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE,
            password_salt BLOB NOT NULL, password_hash BLOB NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
        )""")
        for username, password, role in (("admin", admin_password, "admin"), ("reader", reader_password, "user")):
            salt, digest = hash_password(password)
            connection.execute("""INSERT INTO users(username,password_salt,password_hash,role,created_at)
                VALUES(?,?,?,?,?)""", (username, salt, digest, role, "2026-01-01T00:00:00+00:00"))


def expect_error(error, message, callback, *args):
    try:
        callback(*args)
    except error as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {callback.__name__} to fail: {message}")


def run(directory):
    os.environ["RAG_DATA_DIR"] = directory
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["NO_PROXY"] = "*"
    admin_password, original_password, new_password = ("fixture-admin-password", "fixture-reader-old", "fixture-reader-new")
    setup_legacy_users(Path(directory) / "knowledge.db", admin_password, original_password)

    import app

    db = app.service.db
    admin, *_ = app.login("admin", admin_password)
    reader, *_ = app.login("reader", original_password)
    reader_id = reader["id"]
    assert reader["login_version"] == 0
    _, choices = app.refresh_users(admin)
    assert choices["choices"] == [("reader · 启用", reader_id)]
    session_id = "saved-session"
    question, answer = "报销限额？", "每晚不超过 450 元 [1]。"
    sources = [{"filename": "employee_handbook.md", "content": "每晚不超过 450 元。", "citation": 1}]
    db.log_question(reader_id, session_id, question, answer, 12, [], sources)

    expect_error(app.gr.Error, "仅限管理员", app.disable_user, reader_id, reader)
    expect_error(app.gr.Error, "仅限管理员", app.reset_user_password, reader_id, new_password, reader)
    expect_error(app.gr.Error, "普通用户", app.disable_user, admin["id"], admin)
    expect_error(app.gr.Error, "普通用户", app.reset_user_password, admin["id"], new_password, admin)

    with patch.object(app.service, "answer", side_effect=AssertionError("Blocked request reached the model")):
        app.disable_user(reader_id, admin)
        expect_error(app.gr.Error, "已停用", app.ask, question, reader, session_id)
        expect_error(app.gr.Error, "已停用", app.open_session, session_id, reader)
        expect_error(app.gr.Error, "用户名或密码不正确", app.login, "reader", original_password)
        app.enable_user(reader_id, admin)
        expect_error(app.gr.Error, "登录状态已失效", app.ask, question, reader, session_id)
        resumed, *_ = app.login("reader", original_password)
        app.enable_user(reader_id, admin)
        assert app.current_user(resumed) == resumed
        expect_error(app.gr.Error, "至少需要 10 个字符", app.reset_user_password, reader_id, "short", admin)
        assert app.current_user(resumed) == resumed
        app.reset_user_password(reader_id, new_password, admin)
        expect_error(app.gr.Error, "登录状态已失效", app.ask, question, resumed, session_id)

    expect_error(app.gr.Error, "用户名或密码不正确", app.login, "reader", original_password)
    renewed, *_ = app.login("reader", new_password)
    assert renewed["id"] == reader_id
    history, evidence, *_ = app.open_session(session_id, renewed)
    assert history[-1]["content"] == answer and "450 元" in evidence
    saved, = db.conversation(reader_id, session_id)
    assert json.loads(saved["snapshot"]) == sources

    app.disable_user(reader_id, admin)
    app.reset_user_password(reader_id, original_password, admin)
    expect_error(app.gr.Error, "用户名或密码不正确", app.login, "reader", original_password)
    app.enable_user(reader_id, admin)
    assert app.login("reader", original_password)[0]["id"] == reader_id
    assert app.login("admin", admin_password)[0] == admin
    print("PASS: legacy migration; admin-only controls; disable/restore; password reset; saved citations")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="rag-accounts-check-") as directory:
        run(directory)
