from __future__ import annotations

import json
import secrets
import uuid
from html import escape
import re
from pathlib import Path

import gradio as gr

from rag_system.config import settings
from rag_system.database import Database
from rag_system.service import KnowledgeService

service = KnowledgeService()
first_admin = service.db.ensure_initial_admin()
if first_admin:
    print("\n首次启动管理员账号")
    print(f"用户名：{first_admin[0]}")
    print(f"初始密码：{first_admin[1]}")
    print("请立即登录并妥善保存密码；初始密码仅显示本次。\n")


def current_user(user: dict | None, *, admin: bool = False) -> dict:
    if not user:
        raise gr.Error("请先登录。")
    row = service.db.user_login_state(user["id"])
    if not row or not row["active"]:
        raise gr.Error("账号已停用，请联系管理员。")
    if row["login_version"] != user.get("login_version"):
        raise gr.Error("登录状态已失效，请退出后重新登录。")
    if admin and row["role"] != "admin":
        raise gr.Error("此操作仅限管理员。")
    return user


def login(username: str, password: str):
    user = service.db.authenticate(username or "", password or "")
    if not user:
        raise gr.Error("用户名或密码不正确。")
    return (user, gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=user["role"] == "admin"),
            f"已登录：{user['username']}（{'管理员' if user['role'] == 'admin' else '普通用户'}）",
            [], "")


def logout():
    return (None, gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False), "尚未登录", [], "", "", "", "", uuid.uuid4().hex,
            [], gr.update(choices=[], value=None), gr.update(choices=[], value=None),
            "", "", gr.update(choices=[], value=None), "", "")


def highlight_evidence(content: str, answer: str, question: str) -> str:
    sentences = [s.strip() for s in re.findall(r"[^。！？；\n]+[。！？；]?", content) if s.strip()]
    if not sentences:
        return escape(content)

    def compact(value):
        return re.sub(r"\s+", "", value).casefold()

    terms = re.findall(r"\d+(?:\.\d+)?\s*(?:元|天|小时|分钟|个月|学时|MB|GB|%)|[A-Za-z][A-Za-z0-9-]{2,}", answer)
    terms = list(dict.fromkeys(compact(term) for term in terms))
    from rag_system.retrieval import tokenize
    query_terms = set(tokenize(question))
    best = max(sentences, key=lambda sentence: (
        sum(term in compact(sentence) for term in terms),
        len(query_terms.intersection(tokenize(sentence))),
    ))
    matches = []
    for term in terms:
        pattern = r"\s*".join(re.escape(char) for char in term)
        if term[0].isdigit():
            pattern = r"(?<![\d.])" + pattern
        matches.extend(match.group() for match in re.finditer(pattern, best, re.I))
    if matches:
        pattern = "|".join(re.escape(match) for match in sorted(set(matches), key=len, reverse=True))
        pieces = re.split(f"({pattern})", best)
        highlighted = "".join(f"**{escape(piece)}**" if i % 2 else escape(piece)
                              for i, piece in enumerate(pieces))
    else:
        highlighted = f"**{escape(best)}**"
    before, _, after = content.partition(best)
    return escape(before) + highlighted + escape(after)


def render_sources(sources: list[dict] | None, answer: str, question: str) -> str:
    if sources is None:
        return "此回答产生于引用快照功能上线前，未保存当时的原文。"
    if not sources:
        return "暂无引用来源。"
    active = {doc["sha256"] for doc in service.db.documents()}
    lines = []
    cited = set(re.findall(r"\[(\d+)\]", answer))
    for i, source in enumerate(sources, 1):
        i = source.get("citation", i)
        if str(i) not in cited:
            continue
        page = f" · 第 {source['page_number']} 页" if source.get("page_number") else ""
        score = source.get("rerank_score")
        metric = f" · 语义相似度 {score:.3f}" if score is not None else ""
        excerpt = highlight_evidence(source["content"], answer, question)
        quote = "\n".join("> " + line for line in excerpt.splitlines())
        version = f" · 版本 `{source['sha256'][:12]}`" if source.get("sha256") else ""
        digest = source.get("sha256")
        status = ("当前资料" if digest in active else "历史快照 · 原版本当前不在库")
        if not digest:
            status = "版本状态未知"
        lines.append(f"**[{i}] {escape(source['filename'])}{page}{metric}**{version} · {status}\n\n{quote}")
    return "\n\n".join(lines)


def ask(question: str, user: dict, session_id: str):
    current_user(user)
    if not (question or "").strip():
        return gr.skip(), gr.skip(), "请输入问题。", gr.skip(), gr.skip(), gr.skip()
    history, evidence_history = conversation_view(user["id"], session_id)
    history.append({"role": "user", "content": question.strip()})
    try:
        answer, sources, usage, elapsed = service.answer(
            question.strip(), user, session_id, history[:-1]
        )
    except Exception as exc:
        raise gr.Error(f"问答失败：{exc}") from exc
    history.append({"role": "assistant", "content": answer})
    stats = f"本次耗时：{elapsed} ms"
    if usage:
        stats += f" · Token：输入 {usage.get('input_tokens', 0)} / 输出 {usage.get('output_tokens', 0)}"
    evidence = render_sources(sources, answer, question)
    turns = [*evidence_history, (question.strip(), answer, sources)]
    choices = [(f"{i + 1}. {turn[0][:56]}", i) for i, turn in enumerate(turns)]
    return history, evidence, stats, turns, gr.update(choices=choices, value=len(turns) - 1), session_choices(user, session_id)


def session_choices(user, selected=None):
    current_user(user)
    rows = service.db.sessions(user["id"])
    return gr.update(choices=[(session_title(r["question"]), r["session_id"]) for r in rows], value=selected)


def session_title(question):
    title = re.sub(r"^(?:你好[，,！!\s]*|请问[，,\s]*|我想知道[，,\s]*|帮我查一下[，,\s]*)+", "", " ".join(question.split()))
    title = re.sub(r"(?:是)?(?:多少|什么|几天)[？?。]*$", "", title).rstrip("？?。")
    title = title or " ".join(question.split())
    return title[:20] + ("…" if len(title) > 20 else "")


def conversation_view(user_id, session_id):
    history, turns = [], []
    for row in service.db.conversation(user_id, session_id):
        question, answer = row["question"], row["answer"]
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        sources = None if row["snapshot"] is None else json.loads(row["snapshot"])
        turns.append((question, answer, sources))
    return history, turns


def open_session(selected, user):
    current_user(user)
    history, turns = conversation_view(user["id"], selected)
    if not turns:
        raise gr.Error("无法打开该会话。")
    choices = [(f"{i + 1}. {turn[0][:56]}", i) for i, turn in enumerate(turns)]
    evidence = select_evidence(len(turns) - 1, turns, user)
    return history, evidence, "", turns, gr.update(choices=choices, value=len(turns)-1), selected


def select_evidence(index, turns, user):
    current_user(user)
    if isinstance(index, int) and 0 <= index < len(turns):
        question, answer, sources = turns[index]
        return render_sources(sources, answer, question)
    return "回答来源将在此展示。"


def clear_chat():
    return [], "", "", [], gr.update(choices=[], value=None), uuid.uuid4().hex, gr.update(value=None), ""


def release_rows(user: dict):
    current_user(user, admin=True)
    rows = []
    for release in service.db.knowledge_releases():
        previous = json.loads(release["previous_document"])
        manifest = json.loads(release["manifest"])
        published = next((doc for doc in manifest if doc["id"] == previous["id"]), {})
        rows.append([
            release["id"], release["published_at"], release["username"],
            previous["filename"], published.get("filename", "—"),
            previous["sha256"][:12], published.get("sha256", "—")[:12],
            release["system_version"],
        ])
    return rows


def refresh_documents(user: dict):
    current_user(user, admin=True)
    docs = service.db.documents()
    choices = [(f"{d['filename']} · {d['chunk_count']} 块 · {d['created_at'][:10]}", d["id"]) for d in docs]
    table = "\n".join(
        f"| {d['filename']} | {d['chunk_count']} | {d['uploaded_by'] or '—'} | {d['created_at'][:19]} |"
        for d in docs
    )
    md = "| 文档 | 文本块 | 上传者 | 入库时间 |\n|---|---:|---|---|\n" + table if docs else "知识库尚无文档。"
    return gr.update(choices=choices, value=None), md


def upload_documents(files, user: dict):
    current_user(user, admin=True)
    if not files:
        raise gr.Error("请先选择文件。")
    messages = []
    for item in files:
        path = getattr(item, "name", None) or getattr(item, "path", None) or str(item)
        try:
            messages.append(service.add_document(path, user["id"]))
        except Exception as exc:
            messages.append(f"处理失败：{Path(path).name} — {exc}")
    choices, table = refresh_documents(user)
    return "\n".join(messages), choices, table


def delete_document(doc_id: str, user: dict):
    current_user(user, admin=True)
    if not doc_id:
        raise gr.Error("请选择要删除的文档。")
    result = service.delete_document(doc_id)
    choices, table = refresh_documents(user)
    return result, choices, table


def replace_document(doc_id, file, user):
    current_user(user, admin=True)
    if not doc_id or not file:
        raise gr.Error("请选择要替换的文档并上传新版文件。")
    try:
        release = service.replace_document(doc_id, file, user["id"])
    except Exception as exc:
        raise gr.Error(f"发布失败，未切换文档：{exc}") from exc
    choices, table = refresh_documents(user)
    return f"新版已发布（发布编号 #{release}）。", choices, table, None


def add_user(username: str, password: str, role: str, user: dict):
    current_user(user, admin=True)
    try:
        service.db.create_user(username or "", password or "", "admin" if role == "管理员" else "user")
    except Exception as exc:
        raise gr.Error(f"创建用户失败：{exc}") from exc
    return "用户已创建。", *refresh_users(user), "", ""


def refresh_users(user: dict, selected=None):
    current_user(user, admin=True)
    users = service.db.users()
    choices = [(f"{account['username']} · {'启用' if account['active'] else '停用'}", account["id"])
               for account in users if account["role"] == "user"]
    selected = selected if any(user_id == selected for _, user_id in choices) else None
    return users_markdown(users), gr.update(choices=choices, value=selected)


def manage_user_status(user_id: int | None, user: dict, *, active: bool):
    current_user(user, admin=True)
    if user_id is None:
        raise gr.Error("请先选择普通用户。")
    try:
        changed = service.db.set_user_active(user_id, active)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    state = "启用" if active else "停用"
    message = f"账号已{state}。" if changed else f"账号已经是{state}状态。"
    return message, *refresh_users(user, user_id)


def disable_user(user_id: int | None, user: dict):
    return manage_user_status(user_id, user, active=False)


def enable_user(user_id: int | None, user: dict):
    return manage_user_status(user_id, user, active=True)


def reset_user_password(user_id: int | None, password: str, user: dict):
    current_user(user, admin=True)
    if user_id is None:
        raise gr.Error("请先选择普通用户。")
    try:
        service.db.reset_user_password(user_id, password or "")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return "密码已重置，请通知用户使用新密码重新登录。账号启停状态不变。", ""


def users_markdown(users: list[dict]) -> str:
    rows = "\n".join(
        f"| {u['username']} | {'管理员' if u['role'] == 'admin' else '普通用户'} | {'启用' if u['active'] else '停用'} | {u['created_at'][:19]} |"
        for u in users
    )
    return "| 用户名 | 角色 | 状态 | 创建时间 |\n|---|---|---|---|\n" + rows


def logs_markdown(user: dict) -> str:
    current_user(user, admin=True)
    logs = service.db.recent_logs(100)
    if not logs:
        return "暂无问答日志。"
    rows = "\n".join(
        f"| {l['created_at'][:19]} | {l['username'] or '—'} | {l['question'].replace('|', ' ')} | {l['latency_ms']} ms |"
        for l in logs
    )
    return "| 时间 | 用户 | 问题 | 耗时 |\n|---|---|---|---:|\n" + rows


def seed_demo(user: dict):
    current_user(user, admin=True)
    result = service.seed_demo_documents(user["id"])
    choices, table = refresh_documents(user)
    return result, choices, table


def export_logs(user: dict):
    current_user(user, admin=True)
    path = settings.data_dir / "question_logs_export.jsonl"
    logs = service.db.recent_logs(10000)
    path.write_text("".join(json.dumps(log, ensure_ascii=False) + "\n" for log in logs), encoding="utf-8")
    return str(path)


def library(user):
    current_user(user)
    docs = service.db.documents()
    return "\n".join(f"- {escape(d['filename'])} · {d['chunk_count']} 个片段" for d in docs) or "知识库暂无资料。"


WORKSPACE_CSS = """
#session-choice label, #evidence-choice label, #document-choice label { width: 100%; box-sizing: border-box; cursor: pointer; }
.gradio-container { max-width: 1600px !important; }
#workspace-grid { gap: 16px; align-items: stretch; }
#workspace-grid > .column {
    padding: 18px; border: 1px solid var(--border-color-primary);
    border-radius: 20px; background: var(--background-fill-primary);
}
#library-list, #evidence-list { max-height: 520px; overflow-y: auto; overflow-wrap: anywhere; }
#evidence-list blockquote { padding: 12px; border-radius: 8px; background: var(--background-fill-secondary); }
#evidence-list p { margin-bottom: 10px; }
#login-card { max-width: 440px; margin: 24px auto; }
"""

with gr.Blocks(title="企业知识库问答演示系统") as demo:
    gr.Markdown("# 企业知识库")
    gr.Markdown("单机演示版 · 文档和索引保存在本机；当前使用 DashScope 云端模型，文档片段、问题及检索上下文会发送至模型服务。请勿上传敏感资料。")
    user_state = gr.State(None)
    evidence_history = gr.State([])
    session_state = gr.State(lambda: uuid.uuid4().hex)
    login_status = gr.Markdown("尚未登录")

    with gr.Group(visible=True, elem_id="login-card") as login_group:
        gr.Markdown("## 登录")
        username = gr.Textbox(label="用户名")
        password = gr.Textbox(label="密码", type="password")
        login_btn = gr.Button("登录", variant="primary")

    with gr.Group(visible=False) as workspace:
        with gr.Row():
            gr.Markdown("## 知识问答")
            logout_btn = gr.Button("退出登录", scale=0)
        with gr.Row(elem_id="workspace-grid"):
            with gr.Column(scale=1, min_width=185):
                with gr.Accordion("我的会话", open=False):
                    session_choice = gr.Radio(label="选择会话", show_label=False,
                                              choices=[], elem_id="session-choice")
                gr.Markdown("### 知识库资料")
                library_list = gr.Markdown(elem_id="library-list")
                library_refresh = gr.Button("刷新资料", size="sm")
            with gr.Column(scale=3, min_width=340):
                chatbot = gr.Chatbot(label="对话", height=440)
                question = gr.Textbox(label="问题", placeholder="例如：员工申请年假需要提前多久？", lines=2)
                with gr.Row():
                    ask_btn = gr.Button("提问", variant="primary")
                    clear_btn = gr.Button("新建会话")
                answer_stats = gr.Markdown()
            with gr.Column(scale=2, min_width=260):
                gr.Markdown("### 引用原文")
                with gr.Accordion("选择回答轮次", open=False):
                    evidence_choice = gr.Radio(label="选择回答轮次", show_label=False,
                                               choices=[], elem_id="evidence-choice")
                references = gr.Markdown("回答来源将在此展示。", elem_id="evidence-list")

    with gr.Group(visible=False) as admin_panel:
        gr.Markdown("## 管理员控制台")
        with gr.Tab("文档管理"):
            upload = gr.File(label="添加资料", file_count="multiple", file_types=[".txt", ".md", ".pdf", ".docx"])
            with gr.Row():
                upload_btn = gr.Button("解析并入库", variant="primary")
                seed_btn = gr.Button("导入模拟企业示例资料")
                refresh_btn = gr.Button("刷新列表")
            admin_message = gr.Markdown()
            document_table = gr.Markdown("知识库尚无文档。")
            with gr.Row():
                with gr.Accordion("选择文档", open=False):
                    delete_choice = gr.Radio(label="选择文档", show_label=False,
                                             choices=[], elem_id="document-choice")
                delete_btn = gr.Button("删除文档", variant="stop")
            replacement = gr.File(label="所选文档的新版文件", type="filepath",
                                  file_types=[".txt", ".md", ".pdf", ".docx"])
            replace_btn = gr.Button("替换所选文档并发布")
        with gr.Tab("用户管理") as users_tab:
            with gr.Row():
                new_username = gr.Textbox(label="用户名")
                new_password = gr.Textbox(label="初始密码（至少 10 位）", type="password")
                new_role = gr.Dropdown(label="角色", choices=["普通用户", "管理员"], value="普通用户")
            create_user_btn = gr.Button("创建用户")
            gr.Markdown("#### 管理已有普通账号\n停用或重置密码会使旧登录失效，但不会删除会话和引用。")
            account_to_manage = gr.Dropdown(label="普通用户", choices=[])
            with gr.Row():
                disable_user_btn = gr.Button("停用账号", variant="stop")
                enable_user_btn = gr.Button("恢复账号")
                refresh_users_btn = gr.Button("刷新账号列表")
            with gr.Row():
                reset_password_input = gr.Textbox(label="新密码（至少 10 位）", type="password")
                reset_password_btn = gr.Button("重置所选账号密码")
            user_message = gr.Markdown()
            user_table = gr.Markdown()
        with gr.Tab("发布记录") as release_tab:
            gr.Markdown("最近 100 次文档替换发布（只读，时间为 UTC）。")
            release_refresh = gr.Button("刷新发布记录")
            release_table = gr.Dataframe(
                headers=["编号", "发布时间", "操作人", "替换前", "替换后", "旧版指纹", "新版指纹", "系统版本"],
                interactive=False,
            )
        with gr.Tab("问答日志"):
            log_refresh = gr.Button("刷新日志")
            log_table = gr.Markdown()
            export_btn = gr.Button("导出最近 10000 条日志")
            export_file = gr.File(label="导出文件")

    login_event = login_btn.click(login, [username, password],
                    [user_state, login_group, workspace, admin_panel, login_status, chatbot, answer_stats]).success(
        library, user_state, library_list
    )
    login_event.success(session_choices, user_state, session_choice)
    library_refresh.click(library, user_state, library_list)
    logout_btn.click(logout, outputs=[user_state, login_group, workspace, admin_panel, login_status,
                                      chatbot, answer_stats, references, question, password, session_state,
                                      evidence_history, evidence_choice, session_choice, new_password,
                                      reset_password_input, account_to_manage, user_message, user_table],
                     concurrency_id="chat-actions")
    ask_inputs = [question, user_state, session_state]
    ask_outputs = [chatbot, references, answer_stats, evidence_history, evidence_choice]
    ask_btn.click(ask, ask_inputs, [*ask_outputs, session_choice], concurrency_id="chat-actions").success(
        lambda: "", outputs=question
    )
    question.submit(ask, ask_inputs, [*ask_outputs, session_choice], concurrency_id="chat-actions").success(
        lambda: "", outputs=question
    )
    evidence_choice.input(select_evidence, [evidence_choice, evidence_history, user_state], references)
    clear_btn.click(clear_chat, outputs=[*ask_outputs, session_state, session_choice, question], concurrency_id="chat-actions")
    session_choice.input(open_session, [session_choice, user_state], [*ask_outputs, session_state],
                           concurrency_id="chat-actions")
    upload_btn.click(upload_documents, [upload, user_state], [admin_message, delete_choice, document_table])
    refresh_btn.click(refresh_documents, [user_state], [delete_choice, document_table])
    seed_btn.click(seed_demo, [user_state], [admin_message, delete_choice, document_table])
    delete_btn.click(delete_document, [delete_choice, user_state], [admin_message, delete_choice, document_table])
    replace_btn.click(replace_document, [delete_choice, replacement, user_state],
                      [admin_message, delete_choice, document_table, replacement])
    create_user_btn.click(add_user, [new_username, new_password, new_role, user_state],
                          [user_message, user_table, account_to_manage, new_username, new_password])
    users_tab.select(refresh_users, user_state, [user_table, account_to_manage])
    refresh_users_btn.click(refresh_users, [user_state, account_to_manage], [user_table, account_to_manage])
    account_inputs = [account_to_manage, user_state]
    account_outputs = [user_message, user_table, account_to_manage]
    disable_user_btn.click(disable_user, account_inputs, account_outputs)
    enable_user_btn.click(enable_user, account_inputs, account_outputs)
    reset_password_btn.click(reset_user_password, [account_to_manage, reset_password_input, user_state],
                             [user_message, reset_password_input])
    log_refresh.click(logs_markdown, [user_state], log_table)
    release_tab.select(release_rows, user_state, release_table)
    release_refresh.click(release_rows, user_state, release_table)
    export_btn.click(export_logs, [user_state], export_file)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--reset-admin-password"]:
        from rag_system.auth import hash_password
        new_secret = secrets.token_urlsafe(15)
        salt, digest = hash_password(new_secret)
        with service.db.connect() as conn:
            conn.execute("UPDATE users SET password_salt=?, password_hash=?, login_version=login_version+1 "
                         "WHERE username='admin' AND role='admin'",
                         (salt, digest))
        print(f"admin 新密码：{new_secret}")
    else:
        demo.launch(inbrowser=True, server_name="127.0.0.1", theme=gr.themes.Soft(), css=WORKSPACE_CSS)

