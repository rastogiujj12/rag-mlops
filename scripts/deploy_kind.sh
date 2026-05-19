#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-rag-mlops}"
NAMESPACE="${NAMESPACE:-rag-mlops}"
APP_LABEL="${APP_LABEL:-app=rag-api}"
IMAGE="${IMAGE:-ghcr.io/rastogiujj12/rag-mlops:latest}"
ARTIFACT_DIR="${ARTIFACT_DIR:-runtime-artifacts}"

get_running_pod() {
  kubectl get pod -n "$NAMESPACE" -l "$APP_LABEL" \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || true
}

copy_artifacts_to_pod() {
  local pod="$1"

  echo "Clearing runtime artefact directories..."
  kubectl exec -n "$NAMESPACE" "$pod" -- sh -c 'rm -rf /app/outputs/* /app/artifacts/* /app/data/processed/*'

  echo "Copying outputs..."
  tar -C "$ARTIFACT_DIR/outputs" -cf - . \
    | kubectl exec -i -n "$NAMESPACE" "$pod" -- tar -C /app/outputs -xf -

  echo "Copying processed data..."
  tar -C "$ARTIFACT_DIR/data/processed" -cf - . \
    | kubectl exec -i -n "$NAMESPACE" "$pod" -- tar -C /app/data/processed -xf -

  if [ -d "$ARTIFACT_DIR/artifacts" ]; then
    echo "Copying artefact metadata..."
    tar -C "$ARTIFACT_DIR/artifacts" -cf - . \
      | kubectl exec -i -n "$NAMESPACE" "$pod" -- tar -C /app/artifacts -xf -
  fi
}

echo "Deploying RAG MLOps app to Kind"
echo "Cluster:   $CLUSTER_NAME"
echo "Namespace: $NAMESPACE"
echo "Image:     $IMAGE"
echo

echo "Step 1: Checking required commands..."
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v kind >/dev/null || { echo "kind not found"; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl not found"; exit 1; }
command -v gh >/dev/null || { echo "gh not found"; exit 1; }

echo "Step 2: Checking GitHub authentication..."
gh auth status >/dev/null

echo "Step 3: Ensuring Kind cluster exists..."
if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
  echo "Kind cluster '$CLUSTER_NAME' does not exist."
  echo "Run: make kind-create"
  exit 1
fi

kubectl config use-context "kind-$CLUSTER_NAME"

echo "Step 4: Downloading latest valid CT artefacts..."
bash scripts/download_latest_ct_artifact.sh

echo "Step 5: Pulling and loading Docker image into Kind..."
docker pull "$IMAGE"
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"

echo "Step 6: Applying Kubernetes manifests..."
kubectl apply -k k8s

echo "Step 7: Restarting deployment..."
kubectl rollout restart deployment/rag-api -n "$NAMESPACE"
kubectl rollout status deployment/rag-api -n "$NAMESPACE" --timeout=180s

echo "Step 8: Waiting for running pod..."

POD=""
for i in {1..60}; do
  POD="$(get_running_pod)"
  if [ -n "$POD" ]; then
    echo "Found running pod: $POD"
    break
  fi

  echo "Waiting for running pod..."
  kubectl get pods -n "$NAMESPACE"
  sleep 3
done

if [ -z "$POD" ]; then
  echo "Could not find running rag-api pod."
  kubectl get pods -n "$NAMESPACE"
  exit 1
fi

echo "Using running pod: $POD"

echo "Step 9: Copying CT artefacts into pod runtime volumes..."

test -d "$ARTIFACT_DIR/outputs"
test -d "$ARTIFACT_DIR/data/processed"

set +e
copy_artifacts_to_pod "$POD"
COPY_STATUS=$?
set -e

if [ "$COPY_STATUS" -ne 0 ]; then
  echo "Copy failed, pod may have restarted. Re-selecting current running pod and retrying once..."

  POD=""
  for i in {1..60}; do
    POD="$(get_running_pod)"
    if [ -n "$POD" ]; then
      echo "Found running pod: $POD"
      break
    fi

    echo "Waiting for running pod..."
    kubectl get pods -n "$NAMESPACE"
    sleep 3
  done

  if [ -z "$POD" ]; then
    echo "Could not find running rag-api pod after copy failure."
    kubectl get pods -n "$NAMESPACE"
    exit 1
  fi

  echo "Retrying with pod: $POD"
  copy_artifacts_to_pod "$POD"
fi

echo "Step 10: Verifying artefacts inside pod..."
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/outputs/best_model.json
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/outputs/indices/bm25.pkl
kubectl exec -n "$NAMESPACE" "$POD" -- test -d /app/outputs/indices/dense
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/data/processed/chunks.pkl
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/data/processed/corpus.jsonl

echo "Artefacts copied successfully."
echo

echo "Runtime files inside pod:"
kubectl exec -n "$NAMESPACE" "$POD" -- find /app/outputs -maxdepth 4 -type f | head -30
kubectl exec -n "$NAMESPACE" "$POD" -- find /app/data/processed -maxdepth 2 -type f | head -20

echo
echo "Best model inside pod:"
kubectl exec -n "$NAMESPACE" "$POD" -- cat /app/outputs/best_model.json

echo
echo "Important note:"
echo "If the app loads indexes only at startup, /ready may still report no_index until the pod is restarted with artefacts already mounted."
echo "For this local Kind demo, this script proves artefact hydration. If /ready is still false, use Docker runtime for the full query demo or add hostPath/initContainer next."

echo
echo "Deployment status:"
kubectl get pods -n "$NAMESPACE"
kubectl get svc -n "$NAMESPACE"

echo
echo "To test manually, run:"
echo "kubectl port-forward svc/rag-api -n $NAMESPACE 8000:8000"
echo "curl http://localhost:8000/health"
echo "curl http://localhost:8000/ready"
echo "curl http://localhost:8000/version"