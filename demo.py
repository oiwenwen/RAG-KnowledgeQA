import os
import traceback
from pathlib import Path

# ========== 配置区 ==========
os.environ["DASHSCOPE_API_KEY"] = "sk-ws-H.PRLEDDL.9Z4S.MEUCIQC88_wIinYHmwfZodhQV_W3dlc1CdzUo9y57UKvn5fgvQIgSrHqC_XpjV30cA2JX1jX9a5A-M_dzeaKNAHdt11aZSQ"

# 文档路径
DOC_FILE_PATH = "test.txt"
# 向量库保存路径
VECTOR_STORE_PATH = "faiss_index"

print("【1/6】导入依赖库...")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import DashScopeEmbeddings
    from pypdf import PdfReader
    import docx
    import dashscope
    print("✅ 依赖库导入成功")
except Exception as e:
    print("❌ 依赖导入失败，错误信息：")
    print(traceback.format_exc())
    input("按回车退出...")
    exit()

# ========== 文档读取函数 ==========
def load_document(file_path):
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
        raise ValueError(f"不支持的文件格式：{suffix}，目前支持 .txt / .pdf / .docx")

# ========== 新增：加载或构建向量库 ==========
def load_or_build_vector_store():
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 如果向量库文件已存在，直接加载
    if os.path.exists(VECTOR_STORE_PATH):
        print("📂 找到本地向量库，直接加载...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ 向量库加载完成")
        return vector_store
    
    # 不存在则重新构建并保存
    print("🔄 未找到本地向量库，开始构建...")
    
    print("\n【2/6】读取本地文档...")
    raw_text = load_document(DOC_FILE_PATH)
    print(f"✅ 文档读取成功，文本总长度：{len(raw_text)} 字符")

    print("\n【3/6】文档分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"✅ 分块完成，共 {len(chunks)} 个文本块")

    print("\n【4/6】构建向量库...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # 保存到本地
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ 向量库构建并保存到本地：{VECTOR_STORE_PATH}")
    
    return vector_store

# 加载/构建向量库
vector_store = load_or_build_vector_store()

print("\n【5/6】检索相关内容...")
try:
    query = "上海电机学院电子信息学院有什么专业？"
    related_docs = vector_store.similarity_search(query, k=3)
    print(f"✅ 检索完成，找到 {len(related_docs)} 段相关内容")
except Exception as e:
    print("❌ 检索失败，错误信息：")
    print(traceback.format_exc())
    input("按回车退出...")
    exit()

print("\n【6/6】生成回答（调用大模型）...")
try:
    context = "\n".join([doc.page_content for doc in related_docs])
    
    prompt = f"""请你仅根据下面的上下文内容回答用户的问题。
如果上下文里没有相关信息，就回答"知识库中暂无相关内容"，禁止编造信息。

上下文：
{context}

用户问题：{query}

请给出简洁准确的回答："""

    response = dashscope.Generation.call(
        model="qwen-turbo",
        prompt=prompt,
        result_format="text"
    )

    if response.output is None:
        print(f"❌ API调用失败，错误码：{response.code}")
        print(f"错误信息：{response.message}")
        input("按回车退出...")
        exit()

    answer = response.output.text
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    print("\n=== 回答 ===")
    print(answer)
    
    print("\n=== Token消耗统计 ===")
    print(f"输入Token：{input_tokens}")
    print(f"输出Token：{output_tokens}")
    print(f"本次总共消耗：{total_tokens} Tokens")

    print("\n=== 引用来源片段 ===")
    for i, doc in enumerate(related_docs, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"[{i}] {preview}...")
except Exception as e:
    print("❌ 生成回答失败，错误信息：")
    print(traceback.format_exc())
    input("按回车退出...")
    exit()

print("\n🎉 程序运行完成！")
input("按回车退出...")
