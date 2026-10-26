# -*- coding: utf-8 -*-
"""回归测试：针对 2026-10-26 排查出来的几个真会崩/会静默出错的地方。"""
import pathlib
import sys

import pytest


def test_pdf_page_without_text_does_not_crash(webapp, monkeypatch, tmp_path):
    """扫描件页面的 extract_text() 返回 None，旧代码会 TypeError。

    这是真实场景：PDF 里混一页纯图片，整篇文档就读不进来。
    """

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("第一页正文"), FakePage(None), FakePage("第三页正文")]

    monkeypatch.setattr(webapp, "PdfReader", FakeReader)
    pdf = tmp_path / "mixed.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    text = webapp.load_document(pdf)

    assert "第一页正文" in text
    assert "第三页正文" in text


def test_index_paths_are_absolute(webapp):
    """索引和分块文件必须是绝对路径。

    否则从别的工作目录启动时会在当前目录另建一套，表现为
    "明明建过知识库，问的时候却说请先上传文档"。
    """
    assert pathlib.Path(webapp.VECTOR_STORE_PATH).is_absolute()
    assert pathlib.Path(webapp.CHUNKS_PATH).is_absolute()


def test_chunks_path_next_to_script(webapp):
    """分块文件应落在脚本同目录，而不是工作目录。"""
    assert pathlib.Path(webapp.CHUNKS_PATH).parent == pathlib.Path(webapp.__file__).resolve().parent
