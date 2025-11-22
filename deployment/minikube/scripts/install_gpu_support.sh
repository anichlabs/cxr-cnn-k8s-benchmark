#!/usr/bin/env bash

sudo apt update

# Install NVIDIA driver (you already have 570, adjust if needed)
sudo apt install -y nvidia-driver-570

# Install NVIDIA Container Toolkit for Docker GPU support
sudo apt install -y nvidia-container-toolkit

# Configure Docker runtime to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply GPU runtime changes
sudo systemctl restart docker
