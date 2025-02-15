# Hiring Sentiment Tracker

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced AI-powered system for analyzing job postings and detecting hiring signals. This project uses state-of-the-art NLP techniques and pattern recognition to provide insights into hiring trends and company growth signals.

## Features

- 🎯 **Intelligent Pattern Detection**
  - Multi-algorithm fuzzy matching
  - Support for 6 key signal categories
  - Pattern diversity scoring

- 📊 **Sentiment Analysis**
  - Real-time sentiment scoring
  - Confidence-based analysis
  - Historical trend tracking

- ⚡ **High Performance**
  - ~100ms per job posting
  - Efficient caching system
  - Apple Silicon optimization

## Quick Start

### Prerequisites

- Python 3.8+
- Redis (for caching)
- Node.js 16+ (for frontend)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/UsernameTron/HR-LLM.git
cd HR-LLM
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the services:
```bash
docker-compose up -d  # Starts Redis and monitoring services
python src/main.py   # Starts the main application
```

## Project Structure

```
hiring-sentiment-tracker/
├── src/               # Source code
│   ├── models/       # ML models and algorithms
│   ├── data/         # Data processing and ingestion
│   ├── api/          # API endpoints
│   └── utils/        # Utility functions
├── tests/            # Test suite
├── frontend/         # Vue.js frontend
├── config/           # Configuration files
└── docs/            # Documentation
```

## Development

### Setting up the development environment

1. Install development dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
```

### Running the Frontend

```bash
cd frontend
npm install
npm run serve
```

## Performance Metrics

See our detailed performance analysis in [PHASE_1_RESULTS.md](PHASE_1_RESULTS.md)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Documentation

- [API Documentation](docs/API.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Model Architecture](docs/MODEL.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for the amazing tools and libraries

## Contact

- GitHub: [@UsernameTron](https://github.com/UsernameTron)
- Project Link: [https://github.com/UsernameTron/HR-LLM](https://github.com/UsernameTron/HR-LLM)
