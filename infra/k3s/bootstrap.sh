#!/usr/bin/env bash
# Bootstraps a single-node k3s cluster for local ReRoute development, with
# the NVIDIA device plugin for GPU passthrough to the vLLM pod, and
# kube-prometheus-stack for metrics + Grafana.
set -euo pipefail

echo "==> Installing k3s (single node, no traefik — we use our own ingress later)"
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -

echo "==> Waiting for node to be ready"
sudo k3s kubectl wait --for=condition=Ready node --all --timeout=120s

echo "==> Installing NVIDIA device plugin (requires the NVIDIA container toolkit on the host)"
sudo k3s kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml

echo "==> Adding prometheus-community Helm repo"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

echo "==> Installing kube-prometheus-stack (Prometheus + Grafana)"
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.adminPassword=reroute-dev

echo "==> Importing the ReRoute Grafana dashboard"
sudo k3s kubectl create configmap reroute-dashboard \
  --namespace monitoring \
  --from-file=../grafana/reroute-dashboard.json \
  --dry-run=client -o yaml | sudo k3s kubectl apply -f -

cat <<'EOF'

Done. Next steps:
  1. kubectl create secret generic reroute-provider-keys \
       --from-literal=ANTHROPIC_API_KEY=... --from-literal=NEBIUS_API_KEY=...
  2. helm upgrade --install reroute ../helm/reroute
  3. kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
     (user: admin, password: reroute-dev)
EOF
