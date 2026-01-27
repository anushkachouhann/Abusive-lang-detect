@echo off
echo Starting ML Service...
cd ml-service
python -m pip install -r requirements.txt
python app.py