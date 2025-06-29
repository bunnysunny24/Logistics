@echo off
echo Starting Logistics Pulse Copilot...

:: Check if .env files exist
if not exist "backend\.env" (
    echo ⚠️ Backend .env file not found
    echo Please copy .env.example to backend\.env and configure your API keys
)

if not exist "frontend\.env" (
    echo ⚠️ Frontend .env file not found - creating default...
    echo REACT_APP_API_URL=http://localhost:8000 > frontend\.env
    echo REACT_APP_USER_NAME=user >> frontend\.env
)

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
echo Environment Configuration:
echo - Backend .env: backend\.env
echo - Frontend .env: frontend\.env
echo.
echo Press any key to exit...
pause >nul
