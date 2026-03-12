# UV Installation Guide

This guide uses [uv](https://github.com/astral-sh/uv) for fast, reproducible installation with lock files.

## Quick Start

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project
cd SandboxFusion

# Install all dependencies (main + dev) from lock file
uv sync --python 3.12

# Install Python runtime dependencies
cd runtime/python
uv sync --python 3.10

# Download NLTK data (required for some benchmarks)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Go back to project root
cd ../..

# Create docs directory
mkdir -p docs/build

# Start the service
uv run uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080 --workers 64
```
