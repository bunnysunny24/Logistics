# Logistics Pulse Copilot Startup Script
Write-Host "Starting Logistics Pulse Copilot..." -ForegroundColor Green

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path "data\uploads" | Out-Null

# Function to check if a port is in use
function Test-Port {
    param($Port)
    $tcpConnection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return $tcpConnection.TcpTestSucceeded
}

# Check if ports are available
if (Test-Port 8000) {
    Write-Host "Port 8000 is already in use. Please stop the service using this port." -ForegroundColor Red
    exit 1
}

if (Test-Port 3000) {
    Write-Host "Port 3000 is already in use. Please stop the service using this port." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$PSScriptRoot\backend'; python -m uvicorn main_simple:app --reload --port 8000"
) -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep 3

Write-Host "Starting Frontend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList @(
    "-NoExit", 
    "-Command",
    "cd '$PSScriptRoot\frontend'; npm start"
) -WindowStyle Normal

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Logistics Pulse Copilot is starting up!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend App: http://localhost:3000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
