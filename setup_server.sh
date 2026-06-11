#!/bin/bash
# =============================================
# Setup Script for Ubuntu 22.04 / 24.04 Cloud Server (China Optimized)
# For project: contra_agent on Gitee
# =============================================

set -e  # Exit on any error

echo "=== Starting server setup for China (fast mirrors) ==="

# 1. Change apt sources to Chinese mirrors (auto-detects Ubuntu release + format)
. /etc/os-release
CODENAME="${VERSION_CODENAME:-jammy}"
MAJOR="${VERSION_ID%%.*}"
echo "→ Configuring fast apt mirrors for Ubuntu ${VERSION_ID:-22.04} (${CODENAME})..."

if [ "${MAJOR:-22}" -ge 24 ]; then
    # Ubuntu 24.04+ : deb822 format at /etc/apt/sources.list.d/ubuntu.sources
    sudo cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak 2>/dev/null || true
    sudo tee /etc/apt/sources.list.d/ubuntu.sources > /dev/null << EOF
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
Suites: ${CODENAME} ${CODENAME}-updates ${CODENAME}-backports ${CODENAME}-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
else
    # Ubuntu 22.04 / earlier : traditional one-line format at /etc/apt/sources.list.
    # Remove any stray deb822 file (e.g. left by a prior 24.04-style run) so apt
    # does not pull the wrong (noble) suite.
    sudo rm -f /etc/apt/sources.list.d/ubuntu.sources
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak 2>/dev/null || true
    sudo tee /etc/apt/sources.list > /dev/null << EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME} main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME}-security main restricted universe multiverse
EOF
fi

sudo apt update -qq

# 2. Install essential packages
echo "→ Installing Python and basic tools..."
sudo apt install -y python3-full python3-pip python3-venv git curl wget unzip which

# 3. Clone or update the repository
PROJECT_DIR="$HOME/code/contra_agent"
REPO_URL="https://gitee.com/kaihe_2020/contra_agent.git"

echo "→ Setting up project at $PROJECT_DIR"

if [ -d "$PROJECT_DIR" ]; then
    echo "   Project folder exists, updating..."
    cd "$PROJECT_DIR"
    git config pull.rebase false   # Use merge strategy (safer)
    git pull
else
    echo "   Cloning repository..."
    mkdir -p "$HOME/code"
    cd "$HOME/code"
    git clone "$REPO_URL"
    cd contra_agent
fi

# 4. Clean up any old/broken venv and create a clean one
echo "→ Setting up clean Python virtual environment..."

# Create fresh hidden venv (best practice)
python3 -m venv .venv

# Activate and configure
source .venv/bin/activate

# Set fast PyPI mirror permanently in this venv
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "→ Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "   Warning: requirements.txt not found. Skipping pip install."
fi

echo "→ Installing project in editable mode..."
pip install -e .

# 5. Add convenient alias 'act' and set .venv python as default
echo "→ Adding 'act' alias and setting .venv python as default..."
if ! grep -q "alias act=" "$HOME/.bashrc"; then
    echo 'alias act="cd ~/code/contra_agent && source .venv/bin/activate"' >> "$HOME/.bashrc"
    echo 'alias cenv="cd ~/code/contra_agent"' >> "$HOME/.bashrc"
fi

# Prepend .venv bin to PATH so python3/python always resolves to the venv
if ! grep -q "contra_agent/.venv/bin" "$HOME/.bashrc"; then
    echo 'export PATH="$HOME/code/contra_agent/.venv/bin:$PATH"' >> "$HOME/.bashrc"
fi
# Also add to .profile so it applies to non-interactive (login) shells
if ! grep -q "contra_agent/.venv/bin" "$HOME/.profile"; then
    echo 'export PATH="$HOME/code/contra_agent/.venv/bin:$PATH"' >> "$HOME/.profile"
fi

# Final message
echo "=================================================="
echo "✅ Setup completed successfully!"
echo ""
echo "To start working on your project, run:"
echo "    act"
echo ""
echo "This will take you to the project folder and activate the virtual environment."
echo "After that you can run: python gui_play_human.py  (or whatever your main file is)"
echo ""
echo "Next time you log in, just type 'act' to get ready."
echo "=================================================="