#!/usr/bin/env bash
set -e

BACKUP_DIR="$HOME/backup"
mkdir -p "$BACKUP_DIR/docker"
mkdir -p "$BACKUP_DIR/nvidia"
mkdir -p "$BACKUP_DIR/kube"
mkdir -p "$BACKUP_DIR/k3s"
mkdir -p "$BACKUP_DIR/containerd"

echo "[+] Backing up Docker GPU config..."
[ -f /etc/docker/daemon.json ] && sudo cp /etc/docker/daemon.json "$BACKUP_DIR/docker/"

echo "[+] Backing up NVIDIA container runtime config..."
[ -f /etc/nvidia-container-runtime/config.toml ] && sudo cp /etc/nvidia-container-runtime/config.toml "$BACKUP_DIR/nvidia/"

echo "[+] Backing up containerd configs..."
[ -f /etc/containerd/config.toml ] && sudo cp /etc/containerd/config.toml "$BACKUP_DIR/containerd/"
[ -f /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl ] && sudo cp /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl "$BACKUP_DIR/containerd/config.toml.tmpl"

echo "[+] Backing up K3s configs (if exist)..."
[ -d /etc/rancher/k3s ] && sudo cp -r /etc/rancher/k3s "$BACKUP_DIR/k3s/"

echo "[+] Backing up ~/.kube config..."
[ -f ~/.kube/config ] && cp ~/.kube/config "$BACKUP_DIR/kube/"

echo "[✓] Backup complete!"
echo "Saved to: $BACKUP_DIR"
