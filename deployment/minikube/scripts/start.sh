#!/usr/bin/env bash

# Deletes any previous Minikube cluster.
minikube delete -y

# Starts a fresh Minikube cluster with GPU support.
minikube start \
  --driver=docker \
  --container-runtime=docker \
  --gpus all

# Ensure NVIDIA device plugin is enabled.
minikube addons enable nvidia-device-plugin
