# Deploying the console to Cloud Run

Owner: Niti. Follows S3's guidance: build `index/` offline and ship it in the image, download the e5 weights during the
Docker build (not at container start), and give the service at least 4 GiB of memory.

## 1. Build the artifacts locally (once, after `git pull`)
```
python scripts/prepare.py --split --train --index
```
This creates `data/` (not shipped), `models/triage.joblib` (S2) and `index/` (S3). Re-run `--index` if `src/text.py`
changes: S3's cache key hashes that file, so a stale index is detected and the image build fails on purpose.

## 2. Build and test the image locally
```
docker build -t sentineldesk .
docker run --rm -p 8080:8080 --env-file .env sentineldesk      # http://localhost:8080
```
The build fails if `models/triage.joblib` or an up-to-date `index/` is missing, so the deployed app never runs on stand-ins.

## 3. Deploy
```
gcloud run deploy sentineldesk --source . --region <region> \
  --memory 4Gi --cpu 2 --timeout 300 --session-affinity \
  --min-instances 1 --set-env-vars LLM_PROVIDER=<provider>,LLM_MODEL=<model>
```
- `--memory 4Gi`: the e5 model plus the 33k×768 index needs it (S3).
- `--session-affinity`: Streamlit keeps each user's state (live queue, outage detector) in one container.
- `--min-instances 1`: avoids a cold start (model load) during the demo.
- API keys: use Secret Manager (`--set-secrets ANTHROPIC_API_KEY=<secret>:latest`), never `.env` in the image
  (`.dockerignore` excludes it). Ask S4 which provider/model the demo uses.

## Notes
- `EMBED_DEVICE=cpu` is set in the image. One query takes ~23 ms on CPU (S3's benchmark).
- The image is large (CPU torch + 1.1 GB of weights). Building it the first time takes several minutes.
