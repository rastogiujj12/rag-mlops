#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-rastogiujj12/RAG-MLOPS}"
BRANCH="${BRANCH:-main}"
OUT_DIR="${OUT_DIR:-runtime-artifacts}"

WORKFLOWS=(
  "CT - Smart Pipeline"
)

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Looking for latest valid CT artefact..."
echo "Repository: $REPO"
echo "Branch: $BRANCH"
echo

CANDIDATES_FILE="$(mktemp)"

for WF in "${WORKFLOWS[@]}"; do
  echo "Checking workflow: $WF"

  gh run list \
    --repo "$REPO" \
    --workflow "$WF" \
    --branch "$BRANCH" \
    --status success \
    --limit 10 \
    --json databaseId,createdAt,headSha,name \
    --jq '.[] | [.createdAt, .databaseId, .headSha, .name] | @tsv' \
    >> "$CANDIDATES_FILE" || true
done

if [ ! -s "$CANDIDATES_FILE" ]; then
  echo "ERROR: No successful CT runs found."
  exit 1
fi

# Newest first
while IFS=$'\t' read -r CREATED_AT RUN_ID HEAD_SHA NAME; do  echo
  echo "Trying run:"
  echo "  Workflow: $NAME"
  echo "  Run ID:   $RUN_ID"
  echo "  Commit:   $HEAD_SHA"
  echo "  Created:  $CREATED_AT"

  TMP_DIR="$(mktemp -d)"

  if gh run download "$RUN_ID" --repo "$REPO" --dir "$TMP_DIR"; then
    BEST_MODEL_FILE="$(find "$TMP_DIR" -path "*/outputs/best_model.json" -o -path "$TMP_DIR/outputs/best_model.json" | head -1 || true)"
    CHUNKS_FILE="$(find "$TMP_DIR" -path "*/data/processed/chunks.pkl" -o -path "$TMP_DIR/data/processed/chunks.pkl" | head -1 || true)"
    BM25_FILE="$(find "$TMP_DIR" -path "*/outputs/indices/bm25.pkl" -o -path "$TMP_DIR/outputs/indices/bm25.pkl" | head -1 || true)"
    DENSE_DIR="$(find "$TMP_DIR" -type d -path "*/outputs/indices/dense" | head -1 || true)"

    if [ -n "$BEST_MODEL_FILE" ] && [ -n "$CHUNKS_FILE" ] && [ -n "$BM25_FILE" ] && [ -n "$DENSE_DIR" ]; then
      echo "Valid CT artefact found."

      # Find root folder containing outputs/
      ART_ROOT="$(dirname "$(dirname "$BEST_MODEL_FILE")")"

      rm -rf "$OUT_DIR"
      mkdir -p "$OUT_DIR"
      cp -R "$ART_ROOT"/. "$OUT_DIR"/

      cat > "$OUT_DIR/deployment_artifact_metadata.json" <<EOF
{
  "source_workflow": "$NAME",
  "source_run_id": "$RUN_ID",
  "source_commit": "$HEAD_SHA",
  "created_at": "$CREATED_AT"
}
EOF

      echo
      echo "Downloaded valid CT artefacts to: $OUT_DIR"
      echo "Best model:"
      cat "$OUT_DIR/outputs/best_model.json"
      exit 0
    else
      echo "Artefact from run $RUN_ID is missing required runtime files. Trying older run..."
    fi
  else
    echo "Could not download artefact from run $RUN_ID. Trying older run..."
  fi

  rm -rf "$TMP_DIR"
done < <(sort -r "$CANDIDATES_FILE")

echo "ERROR: No valid CT artefact bundle found."
exit 1