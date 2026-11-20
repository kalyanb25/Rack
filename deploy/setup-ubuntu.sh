#!/usr/bin/env bash
# One-time setup for an Ubuntu box (home server, mini PC, or cloud VM) to run rack.
#   bash deploy/setup-ubuntu.sh
# Installs Docker Engine + Compose, Tailscale, and Python tools. Safe to re-run.
set -euo pipefail

echo "==> Docker Engine + Compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "==> Tailscale (private access from your phone and laptop)"
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "==> Python tools for the rack CLI"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git

if command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
  echo
  echo "NVIDIA GPU detected. For GPU speed in Docker, install the NVIDIA driver and the"
  echo "NVIDIA Container Toolkit: https://docs.ollama.com/docker  (rack up uses the GPU automatically)"
fi

cat <<'EOF'

Done. Next:
  1. Log out and back in (so your user can run docker without sudo)
  2. sudo tailscale up                      # sign in
  3. cd ~/rack && python3 -m venv .venv && . .venv/bin/activate && pip install -e .
  4. rack up
  5. sudo tailscale serve --bg 3000 && sudo tailscale serve --bg --https=8443 11434
EOF
