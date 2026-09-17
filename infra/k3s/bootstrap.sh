#!/usr/bin/env bash
# Bootstraps a single-node k3s cluster for local ReRoute development, with
# GPU passthrough for the vLLM pod and kube-prometheus-stack for
# Prometheus + Grafana (dashboard auto-imported via a labeled ConfigMap).
#
# Order matters here more than it looks: the NVIDIA container runtime is
# configured for *both* Docker and containerd BEFORE k3s is installed, so
# k3s picks up GPU support on its very first boot. Configuring it AFTER
# k3s is already running and then `systemctl restart k3s`-ing to pick up
# the change is what we hit in practice: flannel got stuck in
# "cni plugin not initialized" and the node never came back Ready. A clean
# `k3s-uninstall.sh` + reinstall fixed it immediately, which is why this
# script installs k3s only once, after the runtime is already configured.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "This script needs to run as root (or via sudo) for the apt/systemd/k3s steps." >&2
  exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

echo "==> Installing nvidia-container-toolkit (needs the NVIDIA driver already installed on the host)"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update || true # tolerate unrelated broken repos already on the system
apt-get install -y nvidia-container-toolkit

echo "==> Configuring the NVIDIA runtime for Docker (used for local 'docker run --gpus' testing)"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "==> Pre-registering the NVIDIA runtime for containerd, before k3s exists"
# On recent nvidia-container-toolkit versions this writes a drop-in file
# under /etc/containerd/conf.d/ rather than editing --config's target
# directly. That's fine: k3s's own generated config.toml always has
# `imports = ["/etc/containerd/conf.d/*.toml"]`, so the drop-in is picked
# up automatically once k3s starts.
nvidia-ctk runtime configure --runtime=containerd \
  --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl || true

echo "==> Installing k3s (single node, no traefik — we use our own ingress later)"
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -

echo "==> Waiting for node to be ready"
until k3s kubectl get nodes 2>/dev/null | grep -q " Ready "; do sleep 3; done
k3s kubectl get nodes

echo "==> Setting up kubeconfig for $REAL_USER"
mkdir -p "$REAL_HOME/.kube"
cp /etc/rancher/k3s/k3s.yaml "$REAL_HOME/.kube/config"
chown "$REAL_USER":"$REAL_USER" "$REAL_HOME/.kube/config"
chmod 600 "$REAL_HOME/.kube/config"

echo "==> Installing the NVIDIA device plugin"
k3s kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
# "nvidia" is registered as an *optional* RuntimeClass by k3s, not the
# cluster's default container runtime — the device plugin's own pod needs
# to explicitly opt into it to see the driver library, same as any GPU pod.
k3s kubectl patch daemonset nvidia-device-plugin-daemonset -n kube-system \
  --type=json -p='[{"op":"add","path":"/spec/template/spec/runtimeClassName","value":"nvidia"}]'

echo "==> Adding prometheus-community Helm repo"
sudo -u "$REAL_USER" helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
sudo -u "$REAL_USER" helm repo update

echo "==> Installing kube-prometheus-stack (Prometheus + Grafana)"
sudo -u "$REAL_USER" helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.adminPassword=reroute-dev \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --wait --timeout 5m

echo "==> Importing the ReRoute Grafana dashboard (auto-discovered via the grafana_dashboard=1 label)"
k3s kubectl create configmap reroute-grafana-dashboard \
  --namespace monitoring \
  --from-file=reroute-dashboard.json="$(dirname "$0")/../grafana/reroute-dashboard.json" \
  --dry-run=client -o yaml \
  | k3s kubectl label -f - --local -o yaml grafana_dashboard=1 \
  | k3s kubectl apply -f -

cat <<EOF

Done. Next steps (as $REAL_USER, not root):
  1. kubectl create secret generic reroute-provider-keys \\
       --from-literal=ANTHROPIC_API_KEY=... --from-literal=NEBIUS_API_KEY=...
  2. Build + import the app images (no registry needed for local dev):
       docker build -t reroute-gateway:latest ../../gateway
       docker build -t reroute-router:latest ../../router
       docker save reroute-gateway:latest reroute-router:latest -o /tmp/reroute-images.tar
       sudo k3s ctr images import /tmp/reroute-images.tar
  3. helm upgrade --install reroute ../helm/reroute --set vllm.enabled=true --set serviceMonitor.enabled=true
  4. kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
     (user: admin, password: reroute-dev)
EOF
