@echo off
py -m pip install -r requirements.txt
py -m streamlit run frontend\app.py
pause
