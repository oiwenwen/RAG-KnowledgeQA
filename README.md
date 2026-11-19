# 📚 本地化知识库智能问答系统（RAG）

## 企业知识库演示版

使用 Python 3.11，运行 `python -m pip install -r requirements.txt` 安装依赖。将 `.env.example` 复制为 `.env`，填入 DashScope API Key，再执行 `python app.py`。首次运行在终端显示随机管理员密码。默认仅监听本机 `127.0.0.1`。

如需找回管理员访问权限，关闭应用后运行 `python app.py --reset-admin-password`，在终端查看新密码。

管理员可导入模拟资料、上传或删除文档、安全替换并查看发布记录、管理账号及导出问答日志。普通用户可提问、打开历史会话并核对回答当时的引用。SQLite 保存账号、文档元数据、文本块、向量、会话及引用快照，原文件默认位于 `data/documents`。模型 API 会接收文档片段、问题和上下文，演示请使用公开或模拟资料。

### 账号管理与验收

- 在管理员控制台的「用户管理」中选择普通账号，可停用、恢复或设置至少 10 个字符的新密码。该界面不修改管理员账号。
- 在两个独立浏览器会话中登录管理员和普通用户。保持普通用户页面不刷新，停用后再次提交问题或打开历史会话，应收到停用提示。恢复后，原登录状态仍不可用；退出并重新登录后，可看到原有会话和引用。
- 重置启用账号的密码后，旧登录状态与旧密码均不可用；用新密码登录后会话和引用不变。重置停用账号的密码不会自动恢复该账号。
- 运行 `python -X utf8 evaluation/check_account_management.py` 可离线检查同一套回调和 SQLite 数据迁移。测试使用独立临时目录，不修改现有账号和知识库。

导入模拟资料后，执行 `python evaluation/run_evaluation.py` 比较向量、BM25、混合和混合加重排序。添加 `--generate` 可测回答关键词命中、引用编号有效性和拒答。结果默认保存在 `data/evaluation_results.json`，关键词与引用编号指标需要结合人工核对判断答案是否忠实于原文。

<details>
<summary>历史归档：原预览版文档（不适用于上面的 app.py）</summary>

> **当前为预览版**：核心链路（文档解析 → 混合检索 → 重排序 → 引用溯源 → 流式问答）已完整实现并可运行，界面与功能仍在持续迭代中。

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
├── web_app（预览版）.py      # Web 版：Gradio 界面 + 混合检索 + 多轮问答
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
python "web_app（预览版）.py"
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

## ⚠️ 当前限制（预览版）

- 向量索引采用 **FAISS + 本地文件（pickle）** 的单机方案，不支持多实例横向扩展
- 文档为**上传后整体构建**，文件较多或篇幅较大时构建耗时较长，暂无进度反馈
- 依赖阿里云 DashScope 的**在线**大模型与 Embedding 接口，离线环境不可用
- 暂无用户体系与权限控制，界面按**单用户本地使用**设计
- 暂无自动化测试与 CI

## 🗺️ 后续计划

- [ ] 用户体系与问答历史持久化（数据库存储多会话）
- [ ] 文档**增量入库**（新增/删除单篇文档而不重建整个索引）
- [ ] 文档解析增加进度提示与失败重试
- [ ] 检索效果自动化评测脚本（召回命中率 / 引用准确率）
- [ ] 支持切换本地 Embedding 与本地大模型，实现离线可用
- [ ] 补充界面截图与使用演示

</details>

## 📄 许可

[MIT](LICENSE)
