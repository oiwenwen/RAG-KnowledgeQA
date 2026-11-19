@echo off
rem 启动 RAG 知识库问答系统网页
cd /d "%~dp0"
python "web_app（预览版）.py"
if errorlevel 1 pause
