# Logistics Pulse Copilot

An AI-powered logistics and finance document processing system with real-time anomaly detection and intelligent chat interface.

## Features

- **Document Processing**: Upload and process invoices, shipments, and policy documents
- **AI Chat Interface**: Query documents using natural language with RAG (Retrieval-Augmented Generation)
- **Anomaly Detection**: Real-time detection of anomalies in logistics and financial data
- **Interactive Dashboard**: Visual analytics and statistics for logistics operations
- **Multi-format Support**: Process PDF, CSV, and other document formats

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **Pathway**: Real-time data processing and vector search
- **LangChain**: LLM orchestration and document processing
- **Pydantic**: Data validation and serialization

### Frontend
- **React**: Modern JavaScript framework
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Data visualization
- **Axios**: HTTP client for API calls

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd logistics-pulse-copilot
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r ../requirements.txt
   
   # Copy and configure environment file
   copy .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   
   # Copy and configure environment file (optional)
   copy .env.example .env
   # The frontend .env is optional - defaults will work for local development
   ```

4. **Environment Configuration**
   - **Backend `.env`**: Contains API keys and server configuration
   - **Frontend `.env`**: Contains React app configuration (API URL, user settings)
   - Both files have `.example` templates you can copy from

### Running the Application

#### Option 1: Use the startup script (Windows)
```powershell
# PowerShell
.\start.ps1

# Or Command Prompt
start.bat
```

#### Option 2: Manual startup

**Start Backend:**
```bash
cd backend
python -m uvicorn main_simple:app --reload --port 8000
```

**Start Frontend:**
```bash
cd frontend
npm start
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Usage

1. **Upload Documents**: Use the document uploader to add invoices, shipments, or policy documents
2. **Chat Interface**: Ask questions about your documents using natural language
3. **View Anomalies**: Monitor detected anomalies in the dashboard
4. **Analytics**: Review statistics and trends in your logistics data

## API Endpoints

- `POST /api/query` - Query documents using natural language
- `POST /api/ingest` - Upload and process documents
- `GET /api/anomalies` - Retrieve detected anomalies
- `GET /api/status` - Check system status

## Development

### Backend Development
The backend uses FastAPI with mock data for demonstration. For production use, integrate with:
- Real document processing pipelines
- Vector databases (Pinecone, Weaviate, etc.)
- LLM services (OpenAI, Anthropic, etc.)

### Frontend Development
The React frontend is built with modern practices:
- Component-based architecture
- Custom hooks for API integration
- Responsive design with Tailwind CSS
- Real-time updates and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
