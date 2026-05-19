#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-rag-mlops}"
PROJECT_ROOT="$(pwd)"

echo "Creating Kind cluster config for project:"
echo "$PROJECT_ROOT"

mkdir -p kind
mkdir -p runtime-artifacts/outputs
mkdir -p runtime-artifacts/artifacts
mkdir -p runtime-artifacts/data/processed
mkdir -p models/sentence-transformers

cat > kind/cluster.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4

nodes:
  - role: control-plane
    extraMounts:
      - hostPath: "$PROJECT_ROOT/runtime-artifacts/outputs"
        containerPath: /runtime-artifacts/outputs

      - hostPath: "$PROJECT_ROOT/runtime-artifacts/artifacts"
        containerPath: /runtime-artifacts/artifacts

      - hostPath: "$PROJECT_ROOT/runtime-artifacts/data/processed"
        containerPath: /runtime-artifacts/data/processed

      - hostPath: "$PROJECT_ROOT/models/sentence-transformers"
        containerPath: /models/sentence-transformers
EOF

if kind get clusters | grep -qx "$CLUSTER_NAME"; then
  echo "Kind cluster '$CLUSTER_NAME' already exists."
  echo "Delete and recreate it? [y/N]"
  read -r ANSWER

  if [ "$ANSWER" = "y" ] || [ "$ANSWER" = "Y" ]; then
    kind delete cluster --name "$CLUSTER_NAME"
  else
    echo "Keeping existing cluster."
    echo "Note: changes to extraMounts only apply after recreating the cluster."
    exit 0
  fi
fi

kind create cluster --name "$CLUSTER_NAME" --config kind/cluster.yaml
kubectl config use-context "kind-$CLUSTER_NAME"

echo "Kind cluster ready."
kubectl get nodes