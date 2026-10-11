# 📚 本地化知识库智能问答系统（RAG）

基于**检索增强生成（RAG, Retrieval-Augmented Generation）**技术：上传 TXT / PDF / Word 文档即可构建本地知识库，并通过网页进行**带来源溯源的多轮智能问答**。所有回答严格基于用户提供的文档内容，有效避免大模型凭空编造（幻觉）。

## ✨ 核心特性

- **多格式文档解析**：支持 TXT / PDF / DOCX，支持多文档批量上传
- **混合检索**：向量检索（FAISS 语义匹配）+ 关键词检索（BM25），兼顾语义理解与精确命中
- **查询改写**：结合多轮对话历史，自动补全指代与省略，把追问改写为语义完整的独立检索句
- **重排序（Rerank）**：对候选片段统一计算余弦相似度精排，按阈值过滤低相关内容，取 Top-K
- **来源溯源**：回答中用 `[编号]` 标注来源，界面同步展示引用原文
- **流式输出**：大模型回答逐字流式返回，体验更流畅
- **检索过程可视化**：完整展示查询改写、候选段落匹配度排序等检索链路
- **Token 统计**：实时显示每次问答的 Token 消耗

## 🧱 技术栈

| 模块 | 技术选型 |
| --- | --- |
| 大模型 | 通义千问 Qwen-Turbo（阿里云 DashScope） |
| Embedding 模型 | text-embedding-v2（DashScope） |
| 向量数据库 | FAISS |
| 关键词检索 | BM25（rank-bm25） |
| 应用编排 | LangChain |
| Web 界面 | Gradio |
| 文档解析 | pypdf、python-docx |

## 🏗️ 系统流程

```
文档加载 → 文本分块(RecursiveCharacterTextSplitter) → 向量化入库(FAISS)
   → 用户提问(结合对话历史改写) → 向量 + BM25 混合检索
   → 相似度重排序 / 阈值过滤 → 拼接上下文 Prompt
   → 大模型流式生成 → 带来源引用的回答
```

## 📁 项目结构

```
.
├── demo.py                  # 命令行版：完整 RAG 基础流程演示
├── web_app（初始版）.py      # Web 版：Gradio 界面 + 混合检索 + 多轮问答
├── requirements.txt         # 依赖清单
├── .env.example             # 环境变量示例（复制为 .env 后填入 API Key）
├── test.txt / test.docx     # 测试文档示例
└── README.md
```

> `faiss_index/`、`temp_faiss_index/`、`chunks.pkl` 为程序运行时自动生成的产物，已通过 `.gitignore` 忽略。

## 🚀 快速开始

### 1. 环境准备

要求 Python 3.9+。

```bash
# （可选）创建并激活虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录复制 `.env.example` 并重命名为 `.env`，填入你自己的 DashScope API Key：

```
DASHSCOPE_API_KEY=sk-你的密钥
```

> Key 可在阿里云 DashScope 控制台申请：https://dashscope.console.aliyun.com/

### 3. 运行

**Web 版（推荐）**

```bash
python "web_app（初始版）.py"
```

启动后浏览器会自动打开页面：**上传文档 → 等待知识库构建完成 → 开始提问**。

**命令行演示版**

```bash
python demo.py
```

## ⚙️ 可调参数（位于 web_app 配置区）

- `MAX_HISTORY_ROUND`：参与查询改写的历史对话轮数
- `SIMILARITY_THRESHOLD`：重排序相似度阈值，低于该值的片段会被过滤
- 分块参数：`chunk_size=500`、`chunk_overlap=50`
- 检索数量：向量/BM25 各取 Top3，重排后最终取 Top4

## 🔒 安全说明

- API Key 通过本地 `.env` 文件管理，`.env` 已被 `.gitignore` 忽略，不会进入版本库；
- 向量索引、分块缓存等运行产物同样不上传，克隆后按文档运行即可自动生成。
