# -*- coding: utf-8 -*-
"""pytest 公共夹具：加载 web_app（文件名含全角括号，无法用普通 import，走 importlib）。"""
import importlib.util
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
APP_PATH = PROJECT_ROOT / "web_app（预览版）.py"


def _load_webapp():
    spec = importlib.util.spec_from_file_location("webapp_under_test", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["webapp_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def webapp():
    return _load_webapp()


@pytest.fixture(scope="session")
def tokenize(webapp):
    return webapp.tokenize


@pytest.fixture(scope="session")
def load_document(webapp):
    return webapp.load_document
