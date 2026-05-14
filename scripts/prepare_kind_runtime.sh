#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-$HOME/rag-mlops-runtime}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/rag-model-cache/sentence-transformers}"

mkdir -p "$MODEL_CACHE_ROOT" "$RUNTIME_ROOT/outputs" "$RUNTIME_ROOT/artifacts"

# Copy locally built artefacts into the mounted runtime folder for Kind.
# This keeps the Docker image code-only while the retrieval artefacts remain versionable/replacable.
if [ -d outputs ]; then
  rsync -a --delete outputs/ "$RUNTIME_ROOT/outputs/"
fi
if [ -d artifacts ]; then
  rsync -a --delete artifacts/ "$RUNTIME_ROOT/artifacts/"
fi

cat <<MSG
Prepared Kind runtime folders:
  Model cache: $MODEL_CACHE_ROOT
  Outputs:     $RUNTIME_ROOT/outputs
  Artifacts:   $RUNTIME_ROOT/artifacts

Create Kind cluster with:
  kind create cluster --config kind/cluster.yaml
MSG
