# SupportBot prediction app
#   docker build -t supportbot-app:1.0 .
#   docker run -p 8501:8501 supportbot-app:1.0      ->  http://localhost:8501
# Train first (python src/train.py) so that models/best/ exists.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY samples/ samples/
COPY models/best/ models/best/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
