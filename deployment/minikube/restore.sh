#!/usr/bin/env bash
set -e

BACKUP_DIR="$HOME/backup"

echo "[+] Restoring Docker daemon.json..."
sudo mkdir -p /etc/docker
[ -f "$BACKUP_DIR/docker/daemon.json" ] && sudo cp "$BACKUP_DIR/docker/daemon.json" /etc/docker/

echo "[+] Restoring NVIDIA container runtime..."
sudo mkdir -p /etc/nvidia-container-runtime
[ -f "$BACKUP_DIR/nvidia/config.toml" ] && sudo cp "$BACKUP_DIR/nvidia/config.toml" /etc/nvidia-container-runtime/

echo "[+] Restoring containerd configs..."
sudo mkdir -p /etc/containerd
[ -f "$BACKUP_DIR/containerd/config.toml" ] && sudo cp "$BACKUP_DIR/containerd/config.toml" /etc/containerd/
[ -f "$BACKUP_DIR/containerd/config.toml.tmpl" ] && sudo cp "$BACKUP_DIR/containerd/config.toml.tmpl" /var/lib/rancher/k3s/agent/etc/containerd/

echo "[+] Restoring K3s configs (if needed)..."
[ -d "$BACKUP_DIR/k3s" ] && sudo cp -r "$BACKUP_DIR/k3s" /etc/rancher/

echo "[+] Restoring kubeconfig..."
mkdir -p ~/.kube
[ -f "$BACKUP_DIR/kube/config" ] && cp "$BACKUP_DIR/kube/config" ~/.kube/config

echo "[+] Restarting Docker & containerd..."
sudo systemctl restart docker || true
sudo systemctl restart containerd || true

echo "[✓] Restore complete! Reboot recommended."
