@echo off
echo Starting RAG System Server...
echo The server will run in the background
echo Open http://localhost:8000 in your browser
echo.
echo To stop the server, run: taskkill /f /im python.exe
echo.
start /B python main.py
echo Server started! Check http://localhost:8000
pause
