#!/usr/bin/env bash
set -e

echo "=============================================="
echo " RESTORE GPU RUNTIME + KUBELET + CONTAINERD   "
echo "=============================================="

echo "[1/9] Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
 | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
 | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

echo "[2/9] Configuring containerd NVIDIA runtime..."

sudo mkdir -p /etc/containerd
sudo containerd config default | sudo tee /etc/containerd/config.toml >/dev/null

sudo sed -i "/plugins.\"io.containerd.grpc.v1.cri\".containerd.runtimes/a \
  \ \ \ \ [plugins.\"io.containerd.grpc.v1.cri\".containerd.runtimes.nvidia]\n\
  \ \ \ \ \ \ runtime_type = \"io.containerd.runc.v2\"\n\
  \ \ \ \ \ \ [plugins.\"io.containerd.grpc.v1.cri\".containerd.runtimes.nvidia.options]\n\
  \ \ \ \ \ \ \ \ BinaryName = \"/usr/bin/nvidia-container-runtime\"" \
  /etc/containerd/config.toml

sudo systemctl restart containerd

echo "[3/9] Configuring Docker (optional if not used)..."

sudo mkdir -p /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker || true

echo "[4/9] Validating Docker GPU..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "[5/9] Restoring kubelet GPU settings..."

sudo mkdir -p /etc/systemd/system/kubelet.service.d

cat <<EOF | sudo tee /etc/systemd/system/kubelet.service.d/20-gpu.conf
[Service]
Environment="KUBELET_EXTRA_ARGS=--feature-gates=DevicePlugins=true"
EOF

sudo systemctl daemon-reload
sudo systemctl restart kubelet || true

echo "[6/9] Installing KIND..."
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

echo "[7/9] Restoring KIND GPU cluster config..."

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

echo "[8/9] Creating cluster..."
kind delete cluster || true
kind create cluster --config ~/backup/kind/kind-gpu.yaml

echo "[9/9] Deploying NVIDIA Device Plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.4/nvidia-device-plugin.yml

echo "------------------------------------------------------------------------"
echo " All restored! Now run: kubectl logs gpu-test or deploy your workloads."
echo "------------------------------------------------------------------------"
