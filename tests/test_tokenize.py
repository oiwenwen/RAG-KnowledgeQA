# -*- coding: utf-8 -*-
"""BM25 分词的单元测试。

这些用例锁定了之前修复的关键行为：原先 list(text) 按字符切分，
会导致英文缩写（RAG/FAISS/TDA）检索失效或误召回。改用 jieba 后，
缩写按整词切分并统一小写，中文按词切分。
"""


def test_english_abbreviation_whole_word(tokenize):
    # 缩写必须作为整词保留，而非拆成单字母
    assert tokenize("RAG") == ["rag"]
    assert tokenize("FAISS") == ["faiss"]
    assert tokenize("TDA") == ["tda"]


def test_case_insensitive(tokenize):
    # 大小写归一：查询 "rag" 与文档里的 "RAG" 应匹配
    assert tokenize("rag") == tokenize("RAG") == ["rag"]


def test_hyphenated_english(tokenize):
    # RAG-KnowledgeQA -> rag / knowledgeqa（连字符拆分；CamelCase 串整体保留）
    tokens = tokenize("RAG-KnowledgeQA")
    assert "rag" in tokens
    assert "knowledgeqa" in tokens


def test_alphanumeric_token(tokenize):
    # BM25 -> bm / 25（数字单独成词）
    tokens = tokenize("BM25")
    assert "bm" in tokens
    assert "25" in tokens


def test_chinese_word_segmentation(tokenize):
    tokens = tokenize("上海电机学院电子信息学院")
    assert "上海" in tokens
    assert "学院" in tokens


def test_mixed_cjk_and_english(tokenize):
    tokens = tokenize("RAG 系统采用 FAISS 向量检索")
    assert "rag" in tokens
    assert "系统" in tokens
    assert "faiss" in tokens
    assert "检索" in tokens


def test_whitespace_skipped(tokenize):
    # 空白应被跳过，不产生空 token
    assert "" not in tokenize("   ")
    assert tokenize("A  B") == ["a", "b"]
