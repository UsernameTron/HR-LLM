# AI-Powered Hiring Sentiment Tracker

## Overview
Advanced AI-driven system for detecting hiring signals before job postings go live. Leverages Metal Performance Shaders (MPS) optimization for Apple M4 Pro hardware.

## Key Features
- Multi-label classification for hiring signals
- Real-time and batch processing capabilities
- MPS-optimized transformer models
- Intelligent caching system
- API integration with major data providers
- Drift detection and anomaly tracking
- Explainable AI outputs (SHAP/LIME)

## Hardware Optimization
- Optimized for Apple M4 Pro
- Leverages 18-core GPU with Metal 3 support
- Utilizes 48GB unified memory for efficient batch processing
- NVMe SSD caching for embeddings and frequent queries

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Project Structure
```
hiring-sentiment-tracker/
├── src/
│   ├── core/           # Core ML models and utilities
│   ├── data/           # Data ingestion and processing
│   ├── api/            # API endpoints and services
│   ├── cache/          # Caching mechanisms
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── config/             # Configuration files
├── notebooks/          # Development notebooks
└── scripts/            # Utility scripts
```

## Development
- Uses FastAPI for API endpoints
- Kafka for durable data ingestion
- Redis Streams for low-latency processing
- Automated testing with pytest
- Code formatting with black and isort

## API Integration
Supports multiple data sources:
- NewsAPI
- GDELT
- LinkedIn Insights
- Custom data providers

## Performance Monitoring
- Real-time performance metrics
- Drift detection
- Anomaly tracking
- Resource utilization monitoring

## License
Proprietary - All rights reserved
