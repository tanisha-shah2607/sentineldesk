# Owner: Niti. Container for the SentinelDesk console (Cloud Run or any Docker host).
# Build AFTER creating the artifacts locally (they are gitignored and shipped in the image):
#   python scripts/prepare.py --split --train --index      -> models/triage.joblib, index/
#   docker build -t sentineldesk .
# See docs/deploy_cloud_run.md. data/ is NOT copied: the app needs only models/ and index/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf EMBED_DEVICE=cpu PORT=8080
WORKDIR /app

# CPU-only torch first (the default wheel bundles CUDA and is several GB larger).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the e5 weights (~1.1 GB) into the image so a cold start doesn't download them (S3's advice).
ARG EMBED_MODEL=intfloat/multilingual-e5-base
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_MODEL'], device='cpu')"

COPY . .
# Fail the build early if the artifacts weren't built (otherwise the app silently runs on stand-ins).
RUN test -f models/triage.joblib || (echo 'models/triage.joblib missing: run python scripts/train_triage.py' && exit 1)
RUN python -c "from src.retrieval import index_ready; assert index_ready(), 'index/ missing or stale: run python scripts/build_index.py'"

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.runOnSave=false"]
