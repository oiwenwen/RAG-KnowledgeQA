# -*- coding: utf-8 -*-
"""文档解析（load_document）的单元测试，覆盖 txt / docx 两种格式。"""
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_load_txt(load_document):
    path = str(PROJECT_ROOT / "test.txt")
    text = load_document(path)
    assert isinstance(text, str)
    assert len(text) > 0
    # test.txt 内容为 TDA（拓扑数据分析）介绍
    assert "TDA" in text


def test_load_docx(load_document):
    path = str(PROJECT_ROOT / "test.docx")
    text = load_document(path)
    assert isinstance(text, str)
    assert len(text) > 0


def test_unsupported_extension(load_document):
    # 不支持的格式应抛出 ValueError
    import pytest
    with pytest.raises(ValueError):
        load_document("fake.xlsx")
