@echo off
py -m pip install -r requirements.txt
py -m uvicorn backend.main:app --reload
pause
