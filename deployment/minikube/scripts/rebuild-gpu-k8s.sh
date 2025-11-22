#!/usr/bin/env bash
set -e

echo "[1/8] Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

echo "[2/8] Configuring Docker for GPU runtime..."
sudo mkdir -p /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

sudo systemctl restart docker

echo "[3/8] Testing GPU availability in Docker..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "[4/8] Installing KIND..."
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

echo "[5/8] Writing GPU-enabled KIND cluster config..."
mkdir -p ~/backup/kind
cat <<EOF > ~/backup/kind/kind-gpu.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        feature-gates: "DevicePlugins=true"
- role: worker
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        feature-gates: "DevicePlugins=true"
EOF

echo "[6/8] Creating KIND cluster..."
kind delete cluster || true
kind create cluster --config ~/backup/kind/kind-gpu.yaml

echo "[7/8] Deploying NVIDIA Kubernetes Device Plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.4/nvidia-device-plugin.yml

echo "[8/8] Validating GPU pod..."
kubectl run gpu-test \
  --image=nvidia/cuda:12.4.1-base-ubuntu22.04 \
  --limits='nvidia.com/gpu=1' \
  --restart=Never -- nvidia-smi

echo ""
echo "[✓] Rebuild complete!"
echo "Run: kubectl logs gpu-test"
