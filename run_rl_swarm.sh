#!/bin/bash

# Removed set -euo pipefail for better error handling and process management

# =============================================================================
# RL-Swarm Launcher Script - Enhanced Version with Error Monitoring
# =============================================================================

# Configuration
readonly ROOT="$PWD"
readonly GENRL_TAG="0.1.6"
readonly LOG_DIR="$ROOT/logs"
readonly CONFIG_DIR="$ROOT/configs"

# Environment variables with defaults
export IDENTITY_PATH
export GENSYN_RESET_CONFIG
export CONNECT_TO_TESTNET=true
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120
export SWARM_CONTRACT="0xFaD7C5e93f28257429569B854151A1B8DCD404c2"
export PRG_CONTRACT="0x51D4db531ae706a6eC732458825465058fA23a35"
export HUGGINGFACE_ACCESS_TOKEN="None"
export PORT=${PORT:-3000}
export PRG_GAME=true

# Path configurations
readonly DEFAULT_IDENTITY_PATH="$ROOT/swarm.pem"
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}
DOCKER=${DOCKER:-""}
GENSYN_RESET_CONFIG=${GENSYN_RESET_CONFIG:-""}
CPU_ONLY=${CPU_ONLY:-""}
ORG_ID=${ORG_ID:-""}

# Cloudflared and server variables
TUNNEL_PID=""
TUNNEL_TYPE=""
FORWARDING_URL=""
SERVER_PID=""

# Architecture detection for cloudflared
CF_ARCH=$(uname -m)
case "$CF_ARCH" in
    x86_64) CF_ARCH="amd64" ;;
    aarch64) CF_ARCH="arm64" ;;
    armv7l) CF_ARCH="arm" ;;
    *) CF_ARCH="amd64" ;;
esac

# Color codes
readonly GREEN_TEXT="\033[32m"
readonly BLUE_TEXT="\033[34m"
readonly RED_TEXT="\033[31m"
readonly YELLOW_TEXT="\033[33m"
readonly CYAN_TEXT="\033[36m"
readonly BOLD_TEXT="\033[1m"
readonly RESET_TEXT="\033[0m"

# Aliases for compatibility
readonly GREEN="${GREEN_TEXT}${BOLD_TEXT}"
readonly RED="${RED_TEXT}${BOLD_TEXT}"
readonly YELLOW="${YELLOW_TEXT}${BOLD_TEXT}"
readonly CYAN="${CYAN_TEXT}${BOLD_TEXT}"
readonly BOLD="${BOLD_TEXT}"
readonly NC="${RESET_TEXT}"

# =============================================================================
# Utility Functions
# =============================================================================

# Logging functions
log_info() {
    echo -e "${GREEN_TEXT}[INFO]${RESET_TEXT} $1"
}

log_warn() {
    echo -e "${YELLOW_TEXT}[WARN]${RESET_TEXT} $1"
}

log_error() {
    echo -e "${RED_TEXT}[ERROR]${RESET_TEXT} $1"
}

log_debug() {
    echo -e "${BLUE_TEXT}[DEBUG]${RESET_TEXT} $1"
}

# Legacy echo functions for compatibility
echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

echo_blue() {
    echo -e "$BLUE_TEXT$1$RESET_TEXT"
}

echo_red() {
    echo -e "$RED_TEXT$1$RESET_TEXT"
}

# Initialize directories
init_directories() {
    log_info "Initializing directories..."
    
    local directories=("$LOG_DIR" "$CONFIG_DIR")
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
}

# Docker volume setup
setup_docker_volumes() {
    if [[ -n "$DOCKER" ]]; then
        log_info "Setting up Docker volumes..."
        
        local volumes=(
            "/home/gensyn/rl_swarm/modal-login/temp-data"
            "/home/gensyn/rl_swarm/keys"
            "/home/gensyn/rl_swarm/configs"
            "/home/gensyn/rl_swarm/logs"
        )
        
        for volume in "${volumes[@]}"; do
            if [[ -d "$volume" ]]; then
                sudo chown -R 1001:1001 "$volume"
                log_info "Set ownership for volume: $volume"
            fi
        done
    fi
}

# Enhanced system limits setup
setup_system_limits() {
    log_info "Setting up system limits to prevent resource errors..."
    
    # Increase file descriptor limits
    ulimit -n 65536 2>/dev/null || {
        log_warn "Failed to increase file descriptor limit"
    }
    
    # Increase process limits
    ulimit -u 32768 2>/dev/null || {
        log_warn "Failed to increase process limit"
    }
    
    # Set memory overcommit to prevent allocation failures
    echo 1 | sudo tee /proc/sys/vm/overcommit_memory >/dev/null 2>&1 || {
        log_debug "Could not set memory overcommit (may need sudo)"
    }
    
    log_info "System limits configured"
}

# =============================================================================
# Cloudflared Installation and Management
# =============================================================================

install_cloudflared() {
    if command -v cloudflared >/dev/null 2>&1; then
        echo -e "${GREEN}[✓] Cloudflared is already installed.${NC}"
        return 0
    fi
    
    echo -e "\n${YELLOW}[✓] Installing cloudflared...${NC}"
    
    local cf_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF_ARCH"
    
    if ! wget -q --show-progress "$cf_url" -O cloudflared 2>"$LOG_DIR/cloudflared_download.log"; then
        echo -e "${RED}[✗] Failed to download cloudflared.${NC}"
        [[ -f "$LOG_DIR/cloudflared_download.log" ]] && cat "$LOG_DIR/cloudflared_download.log"
        return 1
    fi
    
    chmod +x cloudflared
    
    if ! sudo mv cloudflared /usr/local/bin/; then
        echo -e "${RED}[✗] Failed to move cloudflared to /usr/local/bin/.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}[✓] Cloudflared installed successfully.${NC}"
    return 0
}

try_cloudflared() {
    echo -e "\n${CYAN}[✓] Trying cloudflared...${NC}"
    
    # Check and install cloudflared
    if ! install_cloudflared; then
        echo -e "${RED}[✗] Failed to install cloudflared. Check $LOG_DIR/cloudflared_download.log for details.${NC}"
        [[ -f "$LOG_DIR/cloudflared_download.log" ]] && cat "$LOG_DIR/cloudflared_download.log"
        return 1
    fi
    
    # Check PORT variable
    if [[ -z "$PORT" ]]; then
        echo -e "${YELLOW}[!] PORT variable is not set. Defaulting to 3000.${NC}"
        PORT=3000
    fi
    
    # Start cloudflared tunnel
    echo -e "${CYAN}[✓] Starting cloudflared tunnel on http://localhost:$PORT...${NC}"
    TUNNEL_TYPE="cloudflared"
    
    local cloudflared_log="$LOG_DIR/cloudflared_output.log"
    cloudflared tunnel --url "http://localhost:$PORT" 2>&1 | tee "$cloudflared_log" &
    TUNNEL_PID=$!
    
    # Wait for URL to appear, max 30 seconds
    local counter=0
    local max_wait=30
    echo -e "${CYAN}[✓] Waiting for cloudflared URL (up to $max_wait seconds)...${NC}"
    
    while [[ $counter -lt $max_wait ]]; do
        if [[ -f "$cloudflared_log" ]]; then
            CLOUDFLARED_URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' "$cloudflared_log" | head -n1)
            if [[ -n "$CLOUDFLARED_URL" ]]; then
                echo -e "${GREEN}[✓] Cloudflared tunnel started: $CLOUDFLARED_URL${NC}"
                FORWARDING_URL="$CLOUDFLARED_URL"
                return 0
            fi
        fi
        sleep 1
        counter=$((counter + 1))
    done
    
    # If URL couldn't be retrieved
    echo -e "${RED}[✗] Failed to retrieve cloudflared URL after $max_wait seconds.${NC}"
    echo -e "${RED}[✗] Check $cloudflared_log for details:${NC}"
    [[ -f "$cloudflared_log" ]] && cat "$cloudflared_log"
    kill $TUNNEL_PID 2>/dev/null || true
    return 1
}

# Cleanup cloudflared tunnel
cleanup_cloudflared() {
    if [[ -n "$TUNNEL_PID" ]]; then
        log_info "Stopping cloudflared tunnel (PID: $TUNNEL_PID)..."
        kill $TUNNEL_PID 2>/dev/null || true
        wait $TUNNEL_PID 2>/dev/null || true
        TUNNEL_PID=""
    fi
}

# Cleanup server process
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        log_info "Stopping modal login server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
    fi
}

# =============================================================================
# Node.js and Yarn Installation
# =============================================================================

install_nodejs() {
    if ! command -v node > /dev/null 2>&1; then
        log_info "Node.js not found. Installing NVM and latest Node.js..."
        
        export NVM_DIR="$HOME/.nvm"
        if [[ ! -d "$NVM_DIR" ]]; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        fi
        
        # Source NVM
        [[ -s "$NVM_DIR/nvm.sh" ]] && \. "$NVM_DIR/nvm.sh"
        [[ -s "$NVM_DIR/bash_completion" ]] && \. "$NVM_DIR/bash_completion"
        
        nvm install node
        log_info "Node.js installed successfully"
    else
        log_info "Node.js is already installed: $(node -v)"
    fi
}

install_yarn() {
    if ! command -v yarn > /dev/null 2>&1; then
        log_info "Installing Yarn..."
        
        # Detect OS and install accordingly
        if grep -qi "ubuntu" /etc/os-release 2> /dev/null || uname -r | grep -qi "microsoft"; then
            log_info "Detected Ubuntu/WSL. Installing Yarn via apt..."
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
            sudo apt update && sudo apt install -y yarn
        else
            log_info "Installing Yarn globally with npm..."
            npm install -g --silent yarn
        fi
        
        log_info "Yarn installed successfully"
    else
        log_info "Yarn is already installed: $(yarn --version)"
    fi
}

# =============================================================================
# Modal Login Server Setup
# =============================================================================

setup_modal_login() {
    log_info "Setting up modal login server..."
    
    cd modal-login || {
        log_error "Failed to change to modal-login directory"
        return 1
    }
    
    # Update environment file
    local env_file="$ROOT/modal-login/.env"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "3s/.*/SWARM_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$env_file"
        sed -i '' "4s/.*/PRG_CONTRACT_ADDRESS=$PRG_CONTRACT/" "$env_file"
    else
        sed -i "3s/.*/SWARM_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$env_file"
        sed -i "4s/.*/PRG_CONTRACT_ADDRESS=$PRG_CONTRACT/" "$env_file"
    fi
    
    # Install dependencies and build (skip if Docker)
    if [[ -z "$DOCKER" ]]; then
        log_info "Installing dependencies and building server..."
        yarn install --immutable
        yarn build > "$LOG_DIR/yarn.log" 2>&1
    fi
    
    # Start server
    log_info "Starting modal login server..."
    yarn start >> "$LOG_DIR/yarn.log" 2>&1 &
    
    SERVER_PID=$!
    log_info "Started server process: $SERVER_PID"
    sleep 5
    
    # Check if userData.json already exists
    local user_data_file="$ROOT/modal-login/temp-data/userData.json"
    
    if [[ ! -f "$user_data_file" ]]; then
        log_info "userData.json not found. Setting up cloudflared tunnel..."
        
        # Try to setup cloudflared tunnel only if userData.json doesn't exist
        if try_cloudflared; then
            log_info "Cloudflared tunnel is available at: $FORWARDING_URL"
        else
            log_warn "Cloudflared tunnel setup failed, falling back to localhost"
        fi
        
        # Open browser
        if [[ -z "$DOCKER" ]]; then
            local url_to_open="http://localhost:3000"
            if [[ -n "$FORWARDING_URL" ]]; then
                url_to_open="$FORWARDING_URL"
            fi
            
            if open "$url_to_open" 2> /dev/null; then
                log_info "Successfully opened $url_to_open in your default browser"
            else
                log_warn "Failed to open $url_to_open. Please open it manually"
            fi
        else
            log_info "Please open http://localhost:3000 in your host browser"
            if [[ -n "$FORWARDING_URL" ]]; then
                log_info "Or use the cloudflared tunnel: $FORWARDING_URL"
            fi
        fi
    else
        log_info "userData.json already exists. Skipping cloudflared setup."
    fi
    
    cd ..
}

# Wait for user data
wait_for_user_data() {
    log_info "Waiting for modal userData.json to be created..."
    
    local user_data_file="modal-login/temp-data/userData.json"
    while [[ ! -f "$user_data_file" ]]; do
        sleep 5
    done
    
    log_info "Found userData.json. Proceeding..."
    
    # Extract ORG_ID
    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' "$user_data_file")
    log_info "Your ORG_ID is set to: $ORG_ID"
    export ORG_ID
}

# Wait for API key activation
wait_for_api_activation() {
    log_info "Waiting for API key to become activated..."
    
    while true; do
        local status
        status=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID" 2>/dev/null || echo "error")
        
        if [[ "$status" == "activated" ]]; then
            log_info "API key is activated! Proceeding..."
            break
        else
            log_info "Waiting for API key to be activated..."
            sleep 5
        fi
    done
}

# =============================================================================
# Python Dependencies Installation
# =============================================================================

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    local packages=(
        "gensyn-genrl==${GENRL_TAG}"
        "reasoning-gym>=0.1.20"
        "hivemind@git+https://github.com/gensyn-ai/hivemind@639c964a8019de63135a2594663b5bec8e5356dd"
    )
    
    for package in "${packages[@]}"; do
        log_info "Installing $package..."
        pip install "$package"
    done
    pip install bitsandbytes accelerate
    pip install colorama
    log_info "Python dependencies installed successfully"
}

# Patch hivemind p2p daemon timeout
patch_hivemind_timeout() {
    log_info "Checking for hivemind p2p daemon patch..."
    
    # Common paths where the file might be located
    local possible_paths=(
        "$ROOT/.venv/lib/python3.12/site-packages/hivemind/p2p/p2p_daemon.py"
        "$HOME/rl-swarm/.venv/lib/python3.12/site-packages/hivemind/p2p/p2p_daemon.py"
        "$(python -c 'import hivemind; print(hivemind.__file__.replace("__init__.py", "p2p/p2p_daemon.py"))' 2>/dev/null || echo '')"
    )
    
    # Try to find the file
    local daemon_file=""
    for path in "${possible_paths[@]}"; do
        if [[ -f "$path" ]]; then
            daemon_file="$path"
            break
        fi
    done
    
    if [[ -n "$daemon_file" ]]; then
        log_info "Found hivemind p2p daemon file: $daemon_file"
        
        # Check if patch is already applied
        if grep -q "startup_timeout: float = 120" "$daemon_file"; then
            log_info "Timeout patch already applied"
        else
            log_info "Applying timeout patch (15s -> 120s)..."
            # Create backup first
            cp "$daemon_file" "$daemon_file.backup"
            
            # Apply patch
            sed -i 's/startup_timeout: float = 15/startup_timeout: float = 120/' "$daemon_file"
            
            # Verify patch was applied
            if grep -q "startup_timeout: float = 120" "$daemon_file"; then
                log_info "Timeout patch applied successfully"
            else
                log_warn "Failed to apply timeout patch"
                # Restore backup if patch failed
                mv "$daemon_file.backup" "$daemon_file"
            fi
        fi
    else
        log_warn "Hivemind p2p daemon file not found. Patch will be skipped."
        log_debug "Searched paths: ${possible_paths[*]}"
    fi
}

# =============================================================================
# Configuration Management
# =============================================================================

setup_config() {
    log_info "Setting up configuration..."
    
    local config_file="$CONFIG_DIR/rg-swarm.yaml"
    local default_config="$ROOT/rgym_exp/config/rg-swarm.yaml"
    
    if [[ -f "$config_file" ]]; then
        if ! cmp -s "$default_config" "$config_file"; then
            if [[ -z "$GENSYN_RESET_CONFIG" ]]; then
                log_warn "Found differences in rg-swarm.yaml. Set GENSYN_RESET_CONFIG to reset to default."
            else
                log_info "Backing up existing config and resetting to default..."
                mv "$config_file" "$config_file.bak"
                cp "$default_config" "$config_file"
            fi
        fi
    else
        log_info "Copying default configuration..."
        cp "$default_config" "$config_file"
    fi
    
    # Set permissions for Docker
    if [[ -n "$DOCKER" ]]; then
        sudo chmod -R 0777 "$CONFIG_DIR"
    fi
}

# =============================================================================
# User Configuration
# =============================================================================

get_user_preferences() {
    echo -en $GREEN_TEXT
    read -p ">> Would you like to push models you train in the RL swarm to the Hugging Face Hub? [y/N] " yn
    echo -en $RESET_TEXT
    yn=${yn:-N} # Default to "N" if the user presses Enter
    case $yn in
        [Yy]*) read -p "Enter your Hugging Face access token: " HUGGINGFACE_ACCESS_TOKEN ;;
        [Nn]*) HUGGINGFACE_ACCESS_TOKEN="None" ;;
        *) echo ">>> No answer was given, so NO models will be pushed to Hugging Face Hub" && HUGGINGFACE_ACCESS_TOKEN="None" ;;
    esac

    echo -en $GREEN_TEXT
    read -p ">> Enter the name of the model you want to use in huggingface repo/name format, or press [Enter] to use the default model. " MODEL_NAME
    echo -en $RESET_TEXT

    # Only export MODEL_NAME if user provided a non-empty value
    if [ -n "$MODEL_NAME" ]; then
        export MODEL_NAME
        echo_green ">> Using model: $MODEL_NAME"
    else
        echo_green ">> Using default model from config"
    fi

    echo -en $GREEN_TEXT
    read -p ">> Would you like your model to participate in the AI Prediction Market? [Y/n] " yn
    if [ "$yn" = "n" ] || [ "$yn" = "N" ]; then
        PRG_GAME=false
        echo_green ">> Playing PRG game: false"
    else
        echo_green ">> Playing PRG game: true"
    fi
    echo -en $RESET_TEXT
}

# =============================================================================
# Enhanced Error Handling and Process Management
# =============================================================================

# Kill existing swarm processes with better detection
kill_existing_swarm() {
    log_warn "Killing existing swarm launcher processes..."
    
    # Kill by process name pattern
    pkill -9 -f rgym_exp.runner.swarm_launcher 2>/dev/null || true
    
    # Kill related Python processes that might be stuck
    pkill -9 -f "hivemind.*dht" 2>/dev/null || true
    pkill -9 -f "python.*swarm" 2>/dev/null || true
    
    # Wait and verify
    sleep 3
    
    # Double check and force kill if needed
    local remaining_pids=$(pgrep -f rgym_exp.runner.swarm_launcher 2>/dev/null || true)
    if [[ -n "$remaining_pids" ]]; then
        log_warn "Force killing remaining processes: $remaining_pids"
        echo "$remaining_pids" | xargs -r kill -9 2>/dev/null || true
    fi
    
    log_info "Process cleanup completed"
}

# Handle interruption signals
handle_interrupt() {
    log_warn "Received interrupt signal (Ctrl+C)..."
    kill_existing_swarm
    cleanup
}

cleanup() {
    log_info "Shutting down trainer..."
    kill_existing_swarm
    cleanup_cloudflared
    cleanup_server
    
    exit 0
}

errnotify() {
    echo_red ">> An error was detected while running rl-swarm. See $ROOT/logs for full logs."
}

# =============================================================================
# Main Execution
# =============================================================================

display_banner() {
    echo -e "\033[38;5;224m"
    cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    From Gensyn - Enhanced Version with Error Monitoring

EOF
    echo -e "$RESET_TEXT"
}

main() {
    # Set up trap for cleanup and interrupt handling
    trap handle_interrupt SIGINT SIGTERM
    trap cleanup EXIT
    trap errnotify ERR
    
    # Initialize
    display_banner
    init_directories
    setup_docker_volumes
    setup_system_limits
    
    # Testnet connection setup
    if [[ "$CONNECT_TO_TESTNET" == "true" ]]; then
        log_info "Setting up testnet connection..."
        echo "Please login to create an Ethereum Server Wallet"
        
        install_nodejs
        install_yarn
        setup_modal_login
        wait_for_user_data
        wait_for_api_activation
    fi
    
    # Install dependencies
    echo_green ">> Getting requirements..."
    install_python_deps
    patch_hivemind_timeout
    setup_config
    
    echo_green ">> Done!"
    
    # Final setup
    echo_green ">> Good luck in the swarm!"
    echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"
    
    # Launch the swarm with enhanced monitoring
    log_info "Launching RL-Swarm with enhanced error monitoring..."
    while true; do
        log_info "Starting swarm launcher..."
        pkill -f "python.*rgym_exp.runner.swarm_launcher" 2>/tmp/swarm_launcher_pkill.log || true
        sleep 1
        # Run the swarm launcher
        python -m rgym_exp.runner.swarm_launcher \
            --config-path "$ROOT/rgym_exp/config" \
            --config-name "rg-swarm.yaml"

        local exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            log_info "Swarm launcher completed successfully"
            break
        else
            log_warn "Swarm launcher exited with code: $exit_code"
            log_info "Restarting in 10 seconds... (Press Ctrl+C to stop)"
            sleep 10
        fi
    done
    
    log_info "Training session completed after $total_restarts restarts"
    
    wait  # Keep script running until Ctrl+C
}

# Execute main function
main "$@"
