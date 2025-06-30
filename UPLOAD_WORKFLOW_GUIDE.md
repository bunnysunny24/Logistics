# Anomaly Detection Dashboard - Upload & Process Workflow

## Overview
The Logistics Pulse Copilot now supports real-time anomaly detection with newly uploaded data. Here's how the complete workflow works:

## Workflow Steps

### 1. Upload New Data Files
- **Frontend**: Use the Document Uploader component
- **API**: POST `/api/upload` with file data
- **Processing**: 
  - Extracts structured data from CSV/PDF files
  - Adds to RAG index for AI queries
  - **Immediately runs anomaly detection** on uploaded data
  - Saves anomalies to persistent storage

### 2. Automatic Anomaly Detection (NEW!)
- **Trigger**: Automatically runs after each successful upload
- **Process**: 
  - Analyzes uploaded data for anomalies
  - Applies business rules and fraud patterns
  - Calculates risk scores and severity levels
  - Updates the anomaly database in real-time

### 4. RAG Pipeline Integration (NEW!)
- **Anomaly Indexing**: All detected anomalies are added to the RAG index
- **Intelligent Queries**: You can now ask questions about detected anomalies
- **Context Awareness**: RAG understands anomaly patterns and relationships
- **Example Queries**:
  - "What high-risk anomalies were detected today?"
  - "Show me suspicious invoices from Problem Supplier Inc"
  - "What are the main anomaly types in recent uploads?"

### 5. Dashboard Updates
- **Real-time**: Dashboard auto-refreshes every 30 seconds
- **Manual**: "Trigger Detection" button forces immediate refresh
- **Data Priority**: Real anomalies take precedence over mock data

## Key Improvements Made

### Backend Enhancements
1. **Upload Processing**: `/api/upload` now immediately processes anomalies
2. **Detection Priority**: Real data replaces mock data when available
3. **Comprehensive Detection**: `/api/detect-anomalies` now processes:
   - All uploaded files (highest priority)
   - Comprehensive data files (fallback)
4. **Persistent Storage**: Anomalies are saved and restored between sessions
5. **RAG Integration**: All anomalies are indexed for intelligent querying

### Frontend Enhancements
1. **Auto-Trigger**: Document upload automatically triggers detection
2. **Auto-Refresh**: Dashboard refreshes every 30 seconds
3. **Enhanced Feedback**: Better success/error messages with anomaly counts
4. **Real-time Updates**: Statistics update immediately after processing

## Testing the Workflow

### Method 1: Complete Workflow Test
```bash
# Make sure backend is running
python start_system.py

# In another terminal, run the complete workflow test
python test_complete_workflow.py
```
This will test the entire pipeline: Upload → Extract → Detect → RAG → Dashboard

### Method 2: Using the Demo Script
```bash
# Make sure backend is running
python start_system.py

# In another terminal, run the demo
python demo_upload_process.py
```

### Method 3: Manual Testing
1. **Start the system**:
   ```bash
   python start_system.py
   ```

2. **Open frontend**: http://localhost:3000

3. **Upload test data**:
   - Go to the Upload section
   - Upload `test_upload_data.csv` (contains 5 test invoices with various anomalies)
   - Watch for success message

4. **Check dashboard**:
   - Navigate to Anomaly Detection Dashboard
   - Should see updated counts
   - Click "Trigger Detection" to force refresh if needed

### Method 4: API Testing
```bash
# Upload file
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@test_upload_data.csv"

# Trigger detection
curl -X POST "http://localhost:8000/api/detect-anomalies"

# Get anomalies
curl "http://localhost:8000/api/anomalies"

# Test RAG queries about anomalies
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "What high-risk anomalies were detected today?"}'
```

## Expected Results

With the test data file (`test_upload_data.csv`), you should see:
- **High Risk**: 2 anomalies
  - INV-TEST-003: Suspicious amount ($99,999.99)
  - INV-TEST-004: Problem supplier with unusual terms
- **Medium Risk**: 1-2 anomalies
  - INV-TEST-001: Large amount deviation
- **Low Risk**: 0-1 anomalies

## File Formats Supported

### CSV Files
- **Invoice data**: Must include `invoice_id` column
- **Shipment data**: Must include `shipment_id` column
- **Auto-detection**: System determines type based on columns

### PDF Files
- **Text extraction**: Uses document processor
- **Pattern matching**: Looks for invoice/shipment patterns
- **Anomaly detection**: Applies rules to extracted data

## Dashboard Features

### Statistics Panel
- **Total Anomalies**: All detected anomalies
- **High Risk**: Risk score ≥ 0.8
- **Medium Risk**: Risk score 0.5-0.79
- **Low Risk**: Risk score < 0.5

### Controls
- **Trigger Detection**: Force re-scan all data
- **Refresh**: Reload current data
- **Filters**: Filter by date, type, severity, etc.

### Auto-Refresh
- **Interval**: Every 30 seconds
- **Condition**: Only when not actively loading/detecting
- **Purpose**: Catch anomalies from new uploads

## Troubleshooting

### Dashboard Shows Old Data
1. Click "Trigger Detection" button
2. Wait for processing to complete
3. Check if new files were uploaded to `data/uploads/`

### No Anomalies After Upload
1. Check file format (CSV must have proper columns)
2. Verify data quality (no empty ID fields)
3. Look at console output for processing errors

### Backend Errors
1. Check if anomaly detector is initialized
2. Verify file permissions in data directories
3. Check logs for specific error messages

## Complete Workflow Verification

The complete workflow now works as follows:

1. **Upload Document/CSV** → 📄 File uploaded to `/data/uploads/`
2. **Extract Data** → 🔍 Text and structured data extracted
3. **Detect Anomalies** → 🚨 Business rules applied, anomalies detected
4. **Update Database** → 💾 Anomalies saved to persistent storage
5. **Index in RAG** → 🧠 Anomalies added to RAG for intelligent queries
6. **Update Dashboard** → 📊 UI shows real-time anomaly statistics
7. **Enable Queries** → 💬 Can ask questions about detected anomalies

### Query Examples After Upload

Once anomalies are detected and indexed, you can ask:
- **"What suspicious activities were found in recent uploads?"**
- **"Show me all high-risk invoice anomalies"**
- **"What patterns do you see in the flagged transactions?"**
- **"Which suppliers have the most anomalies?"**
- **"Explain the risk factors for document TEST-INV-003"**

The system will provide intelligent responses based on the actual detected anomalies.
