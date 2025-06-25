#!/bin/bash
set -e

# HSBC Scam Detection Agent - Development Setup Script
# This script sets up the complete development environment

echo "🚀 HSBC Scam Detection Agent - Development Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if running on supported OS
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python 3.9+
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check Node.js 16+
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version | cut -d'v' -f2)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
        
        if [ "$NODE_MAJOR" -ge 16 ]; then
            print_status "Node.js $NODE_VERSION found"
        else
            print_error "Node.js 16+ required. Found: $NODE_VERSION"
            exit 1
        fi
    else
        print_error "Node.js not found. Please install Node.js 16+"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker found"
    else
        print_error "Docker not found. Please install Docker"
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_status "Docker Compose found"
    else
        print_error "Docker Compose not found. Please install Docker Compose"
        exit 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        print_status "Git found"
    else
        print_error "Git not found. Please install Git"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating project directories..."
    
    directories=(
        "uploads"
        "logs"
        "data/sample_audio"
        "data/training_data"
        "data/mock_data"
        "credentials"
        "static"
        "infrastructure/sql"
        "infrastructure/nginx"
        "infrastructure/prometheus"
        "infrastructure/grafana/dashboards"
        "infrastructure/grafana/datasources"
        "config"
        "tests/fixtures"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

# Create environment files
create_env_files() {
    print_info "Creating environment configuration files..."
    
    # Backend .env file
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# HSBC Scam Detection Agent - Environment Configuration

# API Configuration
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hsbc_fraud_detection
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres123

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Google Cloud Configuration (for production)
# GCP_PROJECT_ID=hsbc-fraud-detection-prod
# GCP_REGION=europe-west2
# GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json

# Agent Configuration
MAX_CONCURRENT_SESSIONS=500
RISK_THRESHOLD_HIGH=80
RISK_THRESHOLD_MEDIUM=40

# Audio Processing
MAX_AUDIO_FILE_SIZE_MB=100
SUPPORTED_AUDIO_FORMATS=.wav,.mp3,.m4a,.flac

# Security
SECRET_KEY=hsbc-scam-detection-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Feature Flags
ENABLE_MOCK_MODE=true
DEMO_MODE=true
ENABLE_LEARNING_AGENT=true
ENABLE_COMPLIANCE_AGENT=true

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO

# CORS Origins
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
EOF
        print_status "Created .env file"
    else
        print_warning ".env file already exists"
    fi
    
    # Frontend .env file
    if [ ! -f "frontend/.env" ]; then
        mkdir -p frontend
        cat > frontend/.env << EOF
# Frontend Environment Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_ENVIRONMENT=development
REACT_APP_VERSION=2.0.0
EOF
        print_status "Created frontend/.env file"
    else
        print_warning "frontend/.env file already exists"
    fi
}

# Create sample audio files (mock)
create_sample_audio() {
    print_info "Creating sample audio files..."
    
    # Create placeholder audio files for demo
    audio_files=(
        "data/sample_audio/investment_scam_live_call.wav"
        "data/sample_audio/romance_scam_live_call.wav"
        "data/sample_audio/impersonation_scam_live_call.wav"
        "data/sample_audio/legitimate_call.wav"
    )
    
    for file in "${audio_files[@]}"; do
        if [ ! -f "$file" ]; then
            # Create a small dummy audio file (1 second of silence)
            # In a real setup, you would have actual audio samples
            echo "Placeholder audio file for demo" > "$file"
            print_status "Created sample audio: $file"
        fi
    done
}

# Create configuration files
create_config_files() {
    print_info "Creating configuration files..."
    
    # Redis configuration
    cat > config/redis.conf << EOF
# Redis configuration for HSBC Scam Detection Agent
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./
EOF
    
    # Nginx configuration
    mkdir -p infrastructure/nginx
    cat > infrastructure/nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }
    
    upstream frontend {
        server frontend:3000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
        
        location /ws/ {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host \$host;
        }
        
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
    }
}
EOF
    
    # Prometheus configuration
    mkdir -p infrastructure/prometheus
    cat > infrastructure/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hsbc-scam-detection-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
      
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
    
    # SQL initialization script
    mkdir -p infrastructure/sql
    cat > infrastructure/sql/init.sql << EOF
-- Initialize HSBC Scam Detection Database

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS fraud_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    customer_id VARCHAR(100),
    agent_id VARCHAR(100),
    risk_score FLOAT DEFAULT 0,
    scam_type VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cases table
CREATE TABLE IF NOT EXISTS fraud_cases (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(100) UNIQUE NOT NULL,
    session_id VARCHAR(100),
    customer_id VARCHAR(100),
    fraud_type VARCHAR(100),
    risk_score FLOAT,
    description TEXT,
    status VARCHAR(50) DEFAULT 'open',
    priority VARCHAR(20) DEFAULT 'medium',
    assigned_to VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_sessions_customer_id ON fraud_sessions(customer_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON fraud_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_cases_status ON fraud_cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON fraud_cases(priority);

-- Insert sample data
INSERT INTO users (username, email, hashed_password) VALUES 
('admin', 'admin@hsbc.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW') -- password: secret
ON CONFLICT (username) DO NOTHING;

COMMIT;
EOF
    
    print_status "Created configuration files"
}

# Setup Python virtual environment
setup_python_env() {
    print_info "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Created Python virtual environment"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Installed Python dependencies"
    else
        print_warning "requirements.txt not found"
    fi
}

# Setup Node.js dependencies
setup_node_env() {
    print_info "Setting up Node.js environment..."
    
    if [ -d "frontend" ]; then
        cd frontend
        
        if [ -f "package.json" ]; then
            npm install
            print_status "Installed Node.js dependencies"
        else
            print_warning "frontend/package.json not found"
        fi
        
        cd ..
    else
        print_warning "frontend directory not found"
    fi
}

# Create Dockerfiles
create_dockerfiles() {
    print_info "Creating Dockerfiles..."
    
    # Backend Dockerfile
    cat > Dockerfile.backend << EOF
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    
    # Frontend Dockerfile
    mkdir -p frontend
    cat > frontend/Dockerfile << EOF
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:3000 || exit 1

# Start application
CMD ["npm", "start"]
EOF
    
    print_status "Created Dockerfiles"
}

# Initialize Git repository
init_git() {
    if [ ! -d ".git" ]; then
        print_info "Initializing Git repository..."
        git init
        
        # Create .gitignore
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log

# Uploads
uploads/
!uploads/.gitkeep

# Credentials
credentials/
*.json
*.pem
*.key

# Database
*.db
*.sqlite3

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build outputs
build/
dist/

# OS
.DS_Store
Thumbs.db

# Docker
.docker/

# Temporary files
tmp/
temp/

# Coverage reports
.coverage
htmlcov/

# Pytest
.pytest_cache/

# MyPy
.mypy_cache/
EOF
        
        git add .gitignore
        git commit -m "Initial commit: Add .gitignore"
        print_status "Initialized Git repository"
    else
        print_warning "Git repository already exists"
    fi
}

# Main setup function
main() {
    echo
    print_info "Starting HSBC Scam Detection Agent setup..."
    echo
    
    check_os
    check_prerequisites
    create_directories
    create_env_files
    create_sample_audio
    create_config_files
    create_dockerfiles
    setup_python_env
    setup_node_env
    init_git
    
    echo
    print_status "Setup completed successfully! 🎉"
    echo
    print_info "Next steps:"
    echo "1. Start the development environment: docker-compose up -d"
    echo "2. Access the frontend: http://localhost:3000"
    echo "3. Access the API docs: http://localhost:8000/docs"
    echo "4. Monitor with Grafana: http://localhost:3001"
    echo "5. View logs with Kibana: http://localhost:5601"
    echo
    print_info "For production deployment, see docs/DEPLOYMENT.md"
    echo
}

# Run main function
main "$@"
