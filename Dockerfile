# AI Truth Engine: CPU image for the web app / API (Hugging Face Spaces compatible).
#   docker build -t truth-engine .
#   docker run -p 7860:7860 truth-engine          # then open http://localhost:7860
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 TQDM_DISABLE=1 HF_HOME=/app/.hf \
    TE_ALLOW_LIVE=1 TE_RATE_LIMIT=10 PORT=7860

# Spaces run containers as uid 1000; give it the app directory
RUN useradd -m -u 1000 app
WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles CUDA and is ~4x larger)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
RUN pip install --no-deps -e . && chown -R app:app /app
USER app

# Bake the pinned models into the image and record their checksums (data/models/MANIFEST.json)
RUN truth-engine setup && python -c "from truth_engine.models_registry import verify; b = verify(); assert not b, b"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c \
  "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"
CMD ["sh", "-c", "truth-engine serve --host=0.0.0.0 --port=${PORT}"]
