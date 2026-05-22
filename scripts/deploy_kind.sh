#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-rag-mlops}"
NAMESPACE="${NAMESPACE:-rag-mlops}"
IMAGE="${IMAGE:-}"

echo "Deploying RAG MLOps app to Kind"
echo "Cluster:   $CLUSTER_NAME"
echo "Namespace: $NAMESPACE"
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

echo "Step 5: Verifying runtime artefacts on host..."
test -f runtime-artifacts/outputs/best_model.json
test -f runtime-artifacts/outputs/indices/bm25.pkl
test -d runtime-artifacts/outputs/indices/dense
test -f runtime-artifacts/data/processed/chunks.pkl
test -f runtime-artifacts/data/processed/corpus.jsonl

echo "Runtime artefacts found:"
find runtime-artifacts -maxdepth 5 -type f | head -40

echo "Step 6: Verifying Kind node can see mounted artefacts..."
docker exec "$CLUSTER_NAME-control-plane" test -f /runtime-artifacts/outputs/best_model.json
docker exec "$CLUSTER_NAME-control-plane" test -f /runtime-artifacts/outputs/indices/bm25.pkl
docker exec "$CLUSTER_NAME-control-plane" test -d /runtime-artifacts/outputs/indices/dense
docker exec "$CLUSTER_NAME-control-plane" test -f /runtime-artifacts/data/processed/chunks.pkl
docker exec "$CLUSTER_NAME-control-plane" test -f /runtime-artifacts/data/processed/corpus.jsonl

echo "Kind node artefacts:"
docker exec "$CLUSTER_NAME-control-plane" find /runtime-artifacts -maxdepth 5 -type f | head -40

echo "Step 7: Determining image from k8s/deployment.yaml..."
if [ -z "$IMAGE" ]; then
  IMAGE="$(grep -m1 "image: ghcr.io/rastogiujj12/rag-mlops:" k8s/deployment.yaml | awk '{print $2}')"
fi

if [ -z "$IMAGE" ]; then
  echo "Could not determine image from k8s/deployment.yaml"
  exit 1
fi

echo "Image: $IMAGE"

echo "Step 8: Pulling/loading Docker image into Kind..."
docker pull "$IMAGE"
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"

echo "Step 9: Applying Kubernetes manifests..."
kubectl apply -k k8s

echo "Step 10: Restarting deployment..."
kubectl rollout restart deployment/rag-api -n "$NAMESPACE"
kubectl rollout status deployment/rag-api -n "$NAMESPACE" --timeout=300s

echo "Step 11: Verifying pod runtime files..."
POD="$(kubectl get pod -n "$NAMESPACE" -l app=rag-api \
  --field-selector=status.phase=Running \
  --sort-by=.metadata.creationTimestamp \
  -o jsonpath='{.items[-1].metadata.name}')"

echo "Using pod: $POD"

kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/outputs/best_model.json
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/outputs/indices/bm25.pkl
kubectl exec -n "$NAMESPACE" "$POD" -- test -d /app/outputs/indices/dense
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/data/processed/chunks.pkl
kubectl exec -n "$NAMESPACE" "$POD" -- test -f /app/data/processed/corpus.jsonl

echo
echo "Best model inside pod:"
kubectl exec -n "$NAMESPACE" "$POD" -- cat /app/outputs/best_model.json

echo
echo "Deployment status:"
kubectl get pods -n "$NAMESPACE"
kubectl get svc -n "$NAMESPACE"

echo
echo "To test manually:"
echo "kubectl port-forward svc/rag-api -n $NAMESPACE 8000:8000"
echo "curl http://localhost:8000/health"
echo "curl http://localhost:8000/ready"
echo "curl http://localhost:8000/version"