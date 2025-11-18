#!/bin/bash
set -euxo pipefail

# ============================
# Configuration
# ============================
GITHUB_REPO_URL="https://github.com/JayChung0258/nlp-multitype-proj.git"
PROJECT_DIR="/home/ubuntu/nlp-multitype-proj"
VENV_DIR="${PROJECT_DIR}/venv"
LOG_DIR="${PROJECT_DIR}/logs"
TRAIN_SCRIPT="train_all_transformers.sh"

# Run commands as ubuntu user
as_ubuntu() {
  sudo -u ubuntu -H bash -lc "$*"
}

# ============================
# 1. System update and essential packages
# ============================
apt-get update -y
apt-get upgrade -y
apt-get install -y python3 python3-venv python3-pip git tmux

# ============================
# 2. Clone or update the project repository
# ============================
if [ ! -d "${PROJECT_DIR}" ]; then
  as_ubuntu "git clone '${GITHUB_REPO_URL}' '${PROJECT_DIR}'"
else
  as_ubuntu "cd '${PROJECT_DIR}' && git pull --rebase"
fi

# ============================
# 3. Create virtual environment and install dependencies
# ============================
if [ ! -d "${VENV_DIR}" ]; then
  as_ubuntu "cd '${PROJECT_DIR}' && python3 -m venv venv"
fi

as_ubuntu "cd '${PROJECT_DIR}' && source venv/bin/activate && pip install --upgrade pip"
as_ubuntu "cd '${PROJECT_DIR}' && source venv/bin/activate && pip install -r requirements.txt"

# ============================
# 4. Create logs directory
# ============================
as_ubuntu "mkdir -p '${LOG_DIR}'"

# ============================
# 5. Launch training inside a tmux session
# ============================
TRAIN_LOG="${LOG_DIR}/train_all_$(date +%Y%m%d_%H%M%S).log"

as_ubuntu "
  cd '${PROJECT_DIR}' && \
  source venv/bin/activate && \
  tmux new -d -s trainer \
    \"bash '${TRAIN_SCRIPT}' 2>&1 | tee '${TRAIN_LOG}'\"
"

# ============================
# 6. Write a boot completion note
# ============================
cat >/home/ubuntu/EC2_BOOT_INFO.txt <<EOF
EC2 bootstrap completed.

Project directory: ${PROJECT_DIR}
Virtualenv: ${VENV_DIR}
Logs: ${LOG_DIR}

A tmux session named 'trainer' is running.
Attach with:   tmux attach -t trainer
Detach with:    Ctrl+B then D

Latest training log:
  ${TRAIN_LOG}
EOF

chown ubuntu:ubuntu /home/ubuntu/EC2_BOOT_INFO.txt

echo "User Data setup finished."
