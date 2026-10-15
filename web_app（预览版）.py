import os
import shutil
from pathlib import Path
import gradio as gr
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv

# ========== 配置区 ==========
# 从本地 .env 文件读取 DASHSCOPE_API_KEY（.env 已被 .gitignore 忽略，不会上传）
load_dotenv()
# ========== 可调配置：默认值 + 环境变量覆盖（.env 里设 RAG_* 即可，无需改代码） ==========
VECTOR_STORE_PATH = os.getenv("RAG_VECTOR_STORE_PATH", "temp_faiss_index")
MAX_HISTORY_ROUND = int(os.getenv("RAG_MAX_HISTORY_ROUND", "2"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen-plus")  # qwen-turbo 已于 2026-10-10 下架
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "gte-rerank-v2")

# ========== 导入依赖 ==========
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import docx
import dashscope
import pickle
import re
import jieba

# ========== 全局缓存 ==========
vector_store_cache = None
bm25_cache = None
all_chunks_cache = None

# ========== 工具函数 ==========
def tokenize(text):
    """BM25 分词：中文走 jieba，英文/数字按单词切分并统一小写。

    原先直接 list(text) 会把 "RAG" 拆成 'R','A','G'，字母级 token 的 IDF 被稀释，
    实测查询 "RAG" 对确实包含该词的段落 BM25 得分为 0（漏检）。
    """
    tokens = []
    for seg in jieba.cut(text):
        seg = seg.strip()
        if not seg:
            continue
        if re.fullmatch(r"[A-Za-z0-9_.\-]+", seg):
            # 纯 ASCII 片段按字母/数字词切分并小写：RAG-KnowledgeQA -> rag, knowledge, qa
            tokens.extend(t.lower() for t in re.findall(r"[A-Za-z]+|\d+", seg))
        else:
            tokens.append(seg)
    return tokens

def load_document(file_path):
    """读取不同格式文档"""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif suffix == ".pdf":
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif suffix == ".docx":
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    else:
        raise ValueError(f"不支持的文件格式：{suffix}")

def build_vector_store(file_obj):
    """上传文档，增量构建/追加知识库。

    首次上传：从零新建向量库；再次上传：只对新文档分块并向量化，
    追加进现有索引，避免重复 embedding 已有文档（BM25 索引基于全量
    chunk 本地重建，不调 API）。按段落内容去重，重复上传不重复入库。
    """
    # 读取已有 chunk（存在则增量追加，不存在则首次构建）
    existing_chunks = []
    if os.path.exists("chunks.pkl"):
        with open("chunks.pkl", "rb") as f:
            existing_chunks = pickle.load(f)
        # 兼容旧版本：纯字符串列表
        if existing_chunks and isinstance(existing_chunks[0], str):
            existing_chunks = [Document(page_content=t) for t in existing_chunks]

    existing_texts = {doc.page_content for doc in existing_chunks}

    # 新上传文档分块（记录来源文件名与段落序号，跳过已存在的重复段落）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    new_chunks = []
    for file_item in file_obj:
        raw_text = load_document(file_item.name)
        pieces = text_splitter.split_text(raw_text)
        fname = Path(file_item.name).name
        for i, piece in enumerate(pieces):
            if piece in existing_texts:
                continue
            new_chunks.append(Document(
                page_content=piece,
                metadata={"source": fname, "chunk_id": i},
            ))

    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    if existing_chunks and os.path.exists(VECTOR_STORE_PATH):
        # 增量：加载现有向量库，只对新 chunk 向量化并追加
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        if new_chunks:
            vector_store.add_documents(new_chunks)
            vector_store.save_local(VECTOR_STORE_PATH)
    else:
        # 首次构建：从零新建
        vector_store = FAISS.from_documents(new_chunks, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)

    # 合并并保存全量 chunk，供 BM25 使用
    all_chunks = existing_chunks + new_chunks
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    # 更新全局缓存
    global vector_store_cache, bm25_cache, all_chunks_cache
    all_chunks_cache = all_chunks
    vector_store_cache = vector_store

    # BM25 索引基于全量 chunk 本地重建（不调 API）
    tokenized_corpus = [tokenize(doc.page_content) for doc in all_chunks]
    bm25_cache = BM25Okapi(tokenized_corpus)

    return f"✅ 知识库构建完成！新增 {len(new_chunks)} 个文本块，当前共 {len(all_chunks)} 块。", []

def answer_question(query, chat_history):
    # 空输入直接返回
    if not query.strip():
        yield chat_history, "", "", "", ""
        return
    
    # ========== 第1步：先显示用户问题 ==========
    chat_history.append({"role": "user", "content": query})
    yield chat_history, "", "", "", ""
    
    # ========== 第2步：检查知识库 ==========
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists("chunks.pkl"):
        chat_history.append({"role": "assistant", "content": "⚠️ 请先上传文档构建知识库！"})
        yield chat_history, "", "", "", ""
        return
    
    # ========== 第2.5步：查询改写（结合历史对话补全语义） ==========
    # 取最近几轮对话用于改写
    history_for_rewrite = chat_history[-MAX_HISTORY_ROUND * 2:]
    rewrite_history_text = ""
    for msg in history_for_rewrite:
        if msg["role"] == "user":
            rewrite_history_text += f"用户：{msg['content']}\n"
        else:
            rewrite_history_text += f"助手：{msg['content']}\n"
    
    # 改写提示词
    rewrite_prompt = f"""请结合对话历史，将用户当前的问题改写为一句完整、独立、无指代省略的检索问句。
要求：
1. 补全省略的主语、指代（如它、此案、这个人等），补充完整上下文信息
2. 只输出改写后的问题本身，不要任何解释、前缀、后缀
3. 保持原问题的核心语义，不要添加额外信息
对话历史：
{rewrite_history_text}
用户当前问题：{query}
改写后的问题："""
    
    # 调用大模型执行改写（非流式）
    # 失败（限流 429 / 网络异常 / 配额不足）时降级为直接用原问题检索，不阻断整个问答
    try:
        rewrite_resp = dashscope.Generation.call(
            model=LLM_MODEL,
            prompt=rewrite_prompt,
            result_format="text",
            stream=False
        )
    except Exception as e:
        rewrite_resp = None
        print("【查询改写异常】", repr(e))
    
    if rewrite_resp is None or getattr(rewrite_resp, "output", None) is None:
        print("【查询改写失败】降级为使用原问题检索；code=%s message=%s" % (
            getattr(rewrite_resp, "code", "exception"),
            getattr(rewrite_resp, "message", ""),
        ))
        rewritten_query = query
    else:
        # 提取改写后的问题，去除首尾空白，空值兜底
        rewritten_query = rewrite_resp.output.text.strip()
        if not rewritten_query:
            rewritten_query = query
    print("【原问题】", query)
    print("【改写后】", rewritten_query)
    
    # ========== 第3步：混合检索上下文 ==========
    global vector_store_cache, bm25_cache, all_chunks_cache
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 向量检索：优先读缓存
    if vector_store_cache is None:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store_cache = vector_store
    else:
        vector_store = vector_store_cache
    vector_docs = vector_store.similarity_search(rewritten_query, k=3)
    
    # BM25检索：优先读缓存
    if bm25_cache is None or all_chunks_cache is None:
        with open("chunks.pkl", "rb") as f:
            all_chunks = pickle.load(f)
        # 兼容旧版本：早期 chunks.pkl 存的是纯字符串列表
        if all_chunks and isinstance(all_chunks[0], str):
            all_chunks = [Document(page_content=t) for t in all_chunks]
        all_chunks_cache = all_chunks
        tokenized_corpus = [tokenize(doc.page_content) for doc in all_chunks]
        bm25_cache = BM25Okapi(tokenized_corpus)
        bm25 = bm25_cache  # 修复：给局部变量赋值
    else:
        all_chunks = all_chunks_cache
        bm25 = bm25_cache
    
    tokenized_query = tokenize(rewritten_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
    bm25_docs = [all_chunks[i] for i in top_bm25_idx]
    
    # 3. 合并去重
    combined_docs = []
    seen_contents = set()
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            combined_docs.append(doc)
    
    # 极端情况：无候选段落直接返回
    if not combined_docs:
        chat_history.append({"role": "assistant", "content": "知识库中暂无相关内容"})
        yield chat_history, "", "", "", ""
        return
    
    # ========== 重排序：rerank 模型精排，失败降级为 embedding 余弦 ==========
    try:
        rerank_resp = dashscope.TextReRank.call(
            model=RERANK_MODEL,
            query=rewritten_query,
            documents=[doc.page_content for doc in combined_docs],
            top_n=min(4, len(combined_docs)),
            return_documents=False,
        )
    except Exception as e:
        rerank_resp = None
        print("【重排序异常】", repr(e))

    if rerank_resp is not None and getattr(rerank_resp, "output", None) is not None \
            and getattr(rerank_resp.output, "results", None):
        # rerank 路径：模型已按相关性排序并截断，无需额外阈值过滤
        rerank_results = rerank_resp.output.results
        scored_docs = [(float(item.relevance_score), combined_docs[item.index]) for item in rerank_results]
        related_docs = [doc for _, doc in scored_docs]
        rank_metric = "rerank相关度"
        rank_pct = False
    else:
        # 降级路径：embedding 余弦相似度 + 阈值过滤
        query_embedding = embeddings.embed_query(rewritten_query)
        query_norm = norm(query_embedding)
        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in combined_docs])
        scored_docs = []
        for doc, doc_embedding in zip(combined_docs, doc_embeddings):
            denominator = query_norm * norm(doc_embedding)
            cos_sim = float(dot(query_embedding, doc_embedding) / denominator) if denominator else 0.0
            scored_docs.append((cos_sim, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        filtered_docs = [(sim, doc) for sim, doc in scored_docs if float(sim) >= SIMILARITY_THRESHOLD]
        related_docs = [doc for sim, doc in filtered_docs[:4]]
        rank_metric = "语义匹配度"
        rank_pct = True

    # 兜底：过滤后无有效段落，直接返回无结果
    if not related_docs:
        chat_history.append({"role": "assistant", "content": "知识库中暂无相关内容"})
        yield chat_history, "", "", "", ""
        return
    
    # ========== 封装检索详情（Markdown美化格式） ==========
    # 1. 构建候选段落排序列表
    rank_content = ""
    for idx, (sim, doc) in enumerate(scored_docs, 1):
        # 截取前60字作为预览，去除换行
        preview_text = doc.page_content[:60].replace("\n", " ").strip() + "..."
        if rank_pct:
            sim_display = round(float(sim) * 100, 1)
            rank_content += f"#### 第 {idx} 名 · {rank_metric} {sim_display}%\n"
        else:
            sim_display = round(float(sim), 4)
            rank_content += f"#### 第 {idx} 名 · {rank_metric} {sim_display}\n"
        rank_content += f"> {preview_text}\n\n"
    
    # 2. 拼接完整的Markdown
    retrieve_detail_md = f"""
### 🔍 检索全流程详情
---
#### 1. 查询改写
| 类型 | 内容 |
| :--- | :--- |
| 原始问题 | {query} |
| 改写后问题 | {rewritten_query} |
---
#### 2. 候选段落语义排序（共 {len(scored_docs)} 段）
{rank_content}
---
✅ 最终筛选 Top **{len(related_docs)}** 段作为生成回答的上下文
"""
    
    # ========== 第4步：构建提示词 ==========
    # 给上下文段落编号
    context = ""
    for i, doc in enumerate(related_docs, 1):
        context += f"[{i}] {doc.page_content}\n\n"
    
    # 拼接历史对话
    history_text = ""
    history_messages = chat_history[-MAX_HISTORY_ROUND * 2:]
    for msg in history_messages:
        if msg["role"] == "user":
            history_text += f"用户：{msg['content']}\n"
        else:
            history_text += f"助手：{msg['content']}\n"
    
   # 最终提示词
    prompt = f"""请你结合对话历史，仅根据下面的上下文内容回答用户的问题。
规则：
1. 所有答案必须100%来自提供的上下文资料，禁止使用任何外部知识，禁止编造上下文没有的信息、人物、情节
2. 每个知识点回答后，必须用 [编号] 标注对应的来源段落，多个来源分别标注，如[1][2]
3. 如果上下文里完全没有相关信息，直接固定回答：知识库中暂无相关内容
4. 严格贴合问题角度回答，不额外发散，不重复无关内容
5. 结合对话历史理解用户的问题，保持回答的连贯性和一致性
对话历史：
{history_text}
上下文：
{context}
用户当前问题：{query}
请给出简洁准确的回答："""
    
    # ========== 第5步：流式调用大模型 ==========
    # 用副本操作，不污染原始对话历史
    temp_chat = chat_history.copy()
    # 预先占位一条空的助手消息
    temp_chat.append({"role": "assistant", "content": ""})

    try:
        responses = dashscope.Generation.call(
            model=LLM_MODEL,
            prompt=prompt,
            result_format="text",
            stream=True
        )
    except Exception as e:
        temp_chat[-1]["content"] = f"❌ 生成失败：{e}"
        yield temp_chat, "", "", "", retrieve_detail_md
        return

    answer = ""
    usage = None
    
    for resp in responses:
        if getattr(resp, "output", None) is None:
            # 限流(429)/配额/网络等错误：给出可读提示，而不是抛 AttributeError
            code = getattr(resp, "code", "")
            message = getattr(resp, "message", "未知错误")
            hint = "（请求过于频繁，请稍后重试）" if "Throttl" in str(code) else ""
            temp_chat[-1]["content"] = f"❌ 调用失败：{message}{hint}"
            yield temp_chat, "", "", "", retrieve_detail_md
            return
        
        # 直接使用最新全量文本，不要累加
        if resp.output.text:
            answer = resp.output.text
        
        # 记录Token统计
        if resp.usage:
            usage = resp.usage
        
        # 实时更新助手消息
        temp_chat[-1]["content"] = answer
        yield temp_chat, "", "", "", retrieve_detail_md
    
    # ========== 第6步：生成完成，返回完整结果 ==========
    # 同步最终对话
    chat_history[:] = temp_chat
    
    # 格式化引用原文
    ref_text = "📄 引用原文（与回答中编号对应）：\n\n"
    for i, doc in enumerate(related_docs, 1):
        src = doc.metadata.get("source", "未知来源")
        cid = doc.metadata.get("chunk_id")
        loc = f"（{src} 第{cid + 1}段）" if isinstance(cid, int) else f"（{src}）"
        ref_text += f"[{i}] {loc} {doc.page_content.strip()}\n\n"
    
    # 格式化Token统计（流式过程未返回 usage 时兜底，避免此处再抛异常）
    if usage is None:
        token_text = "📊 Token消耗统计\n（本次未返回用量信息）"
    else:
        token_text = f"""📊 Token消耗统计
输入Token：{usage.input_tokens}
输出Token：{usage.output_tokens}
本次总计：{usage.total_tokens} Tokens"""
    
    # 最终返回完整结果
    yield chat_history, "", ref_text, token_text, retrieve_detail_md

def clear_knowledge_base():
    """清空整个知识库"""
    global vector_store_cache, bm25_cache, all_chunks_cache
    
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    if os.path.exists("chunks.pkl"):
        os.remove("chunks.pkl")
    
    # 清空内存缓存
    vector_store_cache = None
    bm25_cache = None
    all_chunks_cache = None
    
    # 返回顺序对应前端outputs：chatbot, question_input, ref_output, token_output, upload_status, retrieve_detail_box, file_input
    return [], "", "", "", "⚠️ 知识库已清空，请重新上传文档。", "", None

def clear_chat():
    """清空对话历史和所有显示"""
    return [], "", "", "", ""

# ========== 主题与样式 ==========
THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    # 深色底（黑）+ 墨绿主色
    body_background_fill="#0d1117",
    background_fill_primary="#0d1117",
    background_fill_secondary="#0d1117",
    block_background_fill="#161b22",
    block_border_color="#30363d",
    block_border_width="1px",
    block_radius="8px",
    block_shadow="0 1px 2px rgba(0,0,0,0.4)",
    body_text_color="#e6edf3",
    body_text_color_subdued="#8b949e",
    block_title_text_color="#e6edf3",
    input_background_fill="#161b22",
    input_border_color="#30363d",
    input_placeholder_color="#6e7681",
    button_primary_background_fill="#16a34a",
    button_primary_background_fill_hover="#15803d",
    button_primary_text_color="#ffffff",
    link_text_color="#16a34a",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1500px; margin: 0 auto; }
#app-header { text-align: center; margin-bottom: 0.4em; }
#app-header h1 { margin-bottom: 0.15em; }
#app-subtitle { color: #8b949e; text-align: center; margin-bottom: 1.2em; }
"""

# ========== 构建网页界面 ==========
with gr.Blocks(title="本地化知识库智能问答系统（预览版）") as demo:
    gr.Markdown("# 📚 本地化知识库智能问答系统（预览版）", elem_id="app-header")
    gr.Markdown("上传文档 → 构建知识库 → 带来源溯源的多轮问答 · 支持 TXT / PDF / Word", elem_id="app-subtitle")
    
    with gr.Tabs():
        # ========== Tab 1：知识库管理 ==========
        with gr.Tab("📁 知识库管理"):
            with gr.Row():
                with gr.Column(scale=3):
                    file_input = gr.File(label="上传文档（支持多选，增量追加）", file_types=[".txt", ".pdf", ".docx"], file_count="multiple")
                    upload_status = gr.Textbox(label="构建状态", interactive=False, lines=2)
                with gr.Column(scale=1):
                    gr.Markdown("### 操作")
                    clear_kb_btn = gr.Button("清空知识库", variant="stop")
        
        # ========== Tab 2：知识问答 ==========
        with gr.Tab("💬 知识问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="对话", height=420)
                    with gr.Row():
                        question_input = gr.Textbox(label="输入问题", placeholder="例如：TDA 是什么？", scale=4)
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                        clear_btn = gr.Button("清空对话", scale=1)
                    gr.Markdown("### 📊 消耗统计")
                    token_output = gr.Textbox(label="", interactive=False, lines=4)
                with gr.Column(scale=2):
                    gr.Markdown("### 📄 引用来源")
                    ref_output = gr.Textbox(label="", lines=14)
                    with gr.Accordion("🔍 检索过程详情", open=False):
                        retrieve_detail_box = gr.Markdown(value="")
    
    # 上传文档事件
    file_input.upload(
        fn=build_vector_store,
        inputs=file_input,
        outputs=[upload_status, chatbot]
    )
    
    # 提交问题事件
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, ref_output, token_output, retrieve_detail_box]
    )
    
    # 回车提交
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, ref_output, token_output, retrieve_detail_box]
    )
    
    # 清空知识库事件
    clear_kb_btn.click(
        fn=clear_knowledge_base,
        inputs=[],
        outputs=[chatbot, question_input, ref_output, token_output, upload_status, retrieve_detail_box, file_input]
    )
    
    # 清空对话事件
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, question_input, ref_output, token_output, retrieve_detail_box]
    )

# 启动网页（Gradio 6 起 theme/css 需传 launch 而非 Blocks）
if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=THEME, css=CUSTOM_CSS)
