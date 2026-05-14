#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

curl -fsS "$BASE_URL/health" >/tmp/rag_health.json
cat /tmp/rag_health.json
curl -fsS "$BASE_URL/version" >/tmp/rag_version.json
cat /tmp/rag_version.json

if curl -fsS "$BASE_URL/ready" >/tmp/rag_ready.json; then
  cat /tmp/rag_ready.json
else
  echo "Readiness check failed or index not loaded; /health and /version still responded."
fi
