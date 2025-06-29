# Logistics Pulse Copilot API Testing Guide

## Backend API Endpoints Testing

This guide shows how to test all the API endpoints that are now integrated into the frontend.

### 1. System Health and Status

```powershell
# Check system health
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET

# Get detailed system status
Invoke-WebRequest -Uri "http://localhost:8000/api/status" -Method GET

# Get root API information
Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET
```

### 2. Natural Language Queries

```powershell
# Submit a query about anomalies
$body = @{
    message = "Show me current anomalies"
    context = $null
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/query" -Method POST -ContentType "application/json" -Body $body

# Query about invoices
$body = @{
    message = "How many invoices do we have?"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/query" -Method POST -ContentType "application/json" -Body $body

# Query about shipments
$body = @{
    message = "Show me shipment anomalies"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/query" -Method POST -ContentType "application/json" -Body $body
```

### 3. Anomaly Detection

```powershell
# Get all anomalies
Invoke-WebRequest -Uri "http://localhost:8000/api/anomalies" -Method GET

# Get anomalies with filters
Invoke-WebRequest -Uri "http://localhost:8000/api/anomalies?min_risk_score=0.7&severity=high" -Method GET

# Trigger manual anomaly detection
Invoke-WebRequest -Uri "http://localhost:8000/api/detect-anomalies" -Method POST -ContentType "application/json" -Body "{}"
```

### 4. Risk-Based Holds

```powershell
# Get all risk-based holds
Invoke-WebRequest -Uri "http://localhost:8000/api/risk-holds" -Method GET

# Get active holds only
Invoke-WebRequest -Uri "http://localhost:8000/api/risk-holds?status=active" -Method GET

# Get high-risk holds
Invoke-WebRequest -Uri "http://localhost:8000/api/risk-holds?min_risk_score=0.8" -Method GET
```

### 5. Document Upload

```powershell
# Create a test file
"Test invoice content" | Out-File -FilePath "test_invoice.txt" -Encoding UTF8

# Upload document (using curl for file upload)
curl -X POST "http://localhost:8000/api/upload" -F "file=@test_invoice.txt"

# Alternative using PowerShell (more complex)
$filePath = "test_invoice.txt"
$uri = "http://localhost:8000/api/upload"
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$fileContent = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileBytes)
$boundary = [System.Guid]::NewGuid().ToString()
$bodyTemplate = @"
--{0}
Content-Disposition: form-data; name="file"; filename="{1}"
Content-Type: text/plain

{2}
--{0}--
"@
$body = $bodyTemplate -f $boundary, "test_invoice.txt", $fileContent
Invoke-WebRequest -Uri $uri -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $body
```

### 6. Statistics

```powershell
# Get system statistics
Invoke-WebRequest -Uri "http://localhost:8000/stats" -Method GET
```

### 7. Document Management

```powershell
# Get indexed documents
Invoke-WebRequest -Uri "http://localhost:8000/api/indexed-documents" -Method GET
```

### 8. Memory and Feedback

```powershell
# Submit feedback
$body = @{
    query = "test query"
    answer = "test answer"
    rating = 5
    feedback_text = "Great response!"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/feedback" -Method POST -ContentType "application/json" -Body $body

# Clear conversation memory
Invoke-WebRequest -Uri "http://localhost:8000/api/memory" -Method DELETE
```

## Frontend Integration

The frontend now automatically connects to these endpoints through the new API service:

### React Components Updated:

1. **ChatInterface** - Uses `apiService.submitQuery()` for natural language queries
2. **AnomalyDashboard** - Uses `apiService.getAnomalies()` and `apiService.triggerAnomalyDetection()`
3. **DocumentUploader** - Uses `apiService.uploadDocument()` for file uploads
4. **RiskBasedHoldsPanel** - Uses `apiService.getRiskBasedHolds()` for risk management
5. **SystemStatusPanel** - Uses `apiService.checkHealth()` and `apiService.getSystemStatus()`

### Testing the Frontend:

1. **Start the backend** (if not already running):
   ```powershell
   cd d:\Bunny\logistics-pulse-copilot
   python backend/main_enhanced.py
   ```

2. **Start the frontend**:
   ```powershell
   cd d:\Bunny\logistics-pulse-copilot\frontend
   npm start
   ```

3. **Open browser** to `http://localhost:3000`

4. **Test features**:
   - Check system status panel (top right)
   - Upload a document using the uploader
   - View anomalies in the dashboard
   - Ask questions in the chat interface
   - Check risk-based holds
   - Trigger manual anomaly detection

## Demo Scenario

Perfect demonstration flow for the "before → update → after" requirement:

1. **Before**: Ask "Show me current anomalies" in chat
2. **Update**: Upload a new CSV file with anomalous data or add a file to `data/invoices/`
3. **After**: Ask the same question again - the response should reflect new data

This demonstrates real-time responsiveness to data changes with immediate system updates!
