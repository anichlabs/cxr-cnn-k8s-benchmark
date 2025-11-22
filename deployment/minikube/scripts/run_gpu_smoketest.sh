#!/usr/bin/env bash
set -euo pipefail

POD_NAME="gpu-smoketest"
YAML_PATH="${HOME}/gpu-test.yaml"
CONTEXT_EXPECTED="minikube"

echo "Checking kubectl context..."
CURRENT_CONTEXT=$(kubectl config current-context)

if [[ "${CURRENT_CONTEXT}" != "${CONTEXT_EXPECTED}" ]]; then
  echo "Refusing to run: current context is '${CURRENT_CONTEXT}', expected '${CONTEXT_EXPECTED}'."
  echo "   Use: kubectl config use-context ${CONTEXT_EXPECTED}"
  exit 1
fi

echo "Deleting old pod (if any)..."
kubectl delete pod "${POD_NAME}" --ignore-not-found

echo "Applying ${YAML_PATH}..."
kubectl apply -f "${YAML_PATH}"

echo "Waiting for pod to be Ready..."
kubectl wait --for=condition=Ready "pod/${POD_NAME}" --timeout=120s

echo "Streaming logs from ${POD_NAME}:"
kubectl logs -f "${POD_NAME}"
