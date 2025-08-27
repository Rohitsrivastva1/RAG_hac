Write-Host "Starting RAG System Server..." -ForegroundColor Green
Write-Host "The server will run in the background" -ForegroundColor Yellow
Write-Host "Open http://localhost:8000 in your browser" -ForegroundColor Cyan
Write-Host ""

# Start the server in background
Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python main.py 
} -Name "RAGServer"

Write-Host "Server started in background!" -ForegroundColor Green
Write-Host "Check http://localhost:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop the server, run: Stop-Job -Name 'RAGServer'" -ForegroundColor Red
Write-Host "To see server status, run: Get-Job -Name 'RAGServer'" -ForegroundColor Yellow
Write-Host ""

# Wait a moment for server to start
Start-Sleep -Seconds 3

# Check if server is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Server is running successfully!" -ForegroundColor Green
    Write-Host "🌐 Web Interface: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📖 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Server might still be starting up..." -ForegroundColor Yellow
    Write-Host "Wait a few more seconds and try http://localhost:8000" -ForegroundColor Yellow
}
