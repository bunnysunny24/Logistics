@echo off
echo Starting Logistics Pulse Copilot...

:: Create directories if they don't exist
if not exist "data\uploads" mkdir "data\uploads"

echo Starting Backend Server...
start "Backend" cmd /k "cd /d backend && python -m uvicorn main_simple:app --reload --port 8000"

:: Wait a moment for backend to start
timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "Frontend" cmd /k "cd /d frontend && npm start"

echo.
echo ==============================================
echo Logistics Pulse Copilot is starting up!
echo ==============================================
echo Backend API: http://localhost:8000
echo Frontend App: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo ==============================================
echo.
echo Press any key to exit...
pause >nul
