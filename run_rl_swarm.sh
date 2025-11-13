#!/usr/bin/env bash

# Removed set -euo pipefail for better error handling and process management

# =============================================================================
# RL-Swarm Launcher Script - Code-Gen Enhanced Version
# =============================================================================

# Configuration
readonly ROOT="$PWD"
readonly LOG_DIR="$ROOT/logs"
readonly CONFIG_DIR="$ROOT/configs"

# Environment variables with defaults
export IDENTITY_PATH
export GENSYN_RESET_CONFIG
export CONNECT_TO_TESTNET=true
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120
export SWARM_CONTRACT="0x7745a8FE4b8D2D2c3BB103F8dCae822746F35Da0"
export HUGGINGFACE_ACCESS_TOKEN="None"

# Path configurations
readonly DEFAULT_IDENTITY_PATH="$ROOT/swarm.pem"
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}
DOCKER=${DOCKER:-""}
GENSYN_RESET_CONFIG=${GENSYN_RESET_CONFIG:-""}
CPU_ONLY=${CPU_ONLY:-""}
ORG_ID=${ORG_ID:-""}

# Server variables
SERVER_PID=""

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
    echo "Please login to create an Ethereum Server Wallet"
    
    cd modal-login || {
        log_error "Failed to change to modal-login directory"
        return 1
    }
    
    # Update environment file
    local env_file="$ROOT/modal-login/.env"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "3s/.*/SWARM_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$env_file"
    else
        sed -i "3s/.*/SWARM_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$env_file"
    fi
    
    # Install dependencies and build (skip if Docker)
    if [[ -z "$DOCKER" ]]; then
        log_info "Installing dependencies and building server..."
        yarn install --immutable
        log_info "Building server"
        yarn build > "$LOG_DIR/yarn.log" 2>&1
    fi
    
    # Start server
    log_info "Starting modal login server..."
    yarn start >> "$LOG_DIR/yarn.log" 2>&1 &
    
    SERVER_PID=$!
    log_info "Started server process: $SERVER_PID"
    sleep 5
    
    # Open browser
    if [[ -z "$DOCKER" ]]; then
        if open http://localhost:3000 2> /dev/null; then
            log_info "Successfully opened http://localhost:3000 in your default browser"
        else
            log_warn "Failed to open http://localhost:3000. Please open it manually"
        fi
    else
        log_info "Please open http://localhost:3000 in your host browser"
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
# Ollama Installation
# =============================================================================

install_ollama() {
    # Ollama already running as part of the docker compose file
    if [[ -z "$DOCKER" ]]; then
        echo_green ">> Installing Ollama requires 'sudo' privileges. As an alternative, please use the Docker installation path as described in README.md"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # Install brew if not already installed
            if ! command -v brew > /dev/null 2>&1; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            # Install ollama if not already installed
            if ! command -v ollama > /dev/null 2>&1; then
                log_info "Installing Ollama via Homebrew..."
                brew install ollama
            fi
        else
            # Install ollama if not already installed
            if ! command -v ollama > /dev/null 2>&1; then
                log_info "Installing Ollama..."
                curl -fsSL https://ollama.com/install.sh | sh -s -- -y
            fi
        fi
        
        # Start ollama server if not already running
        if ! ollama list > /dev/null 2>&1; then
            log_info "Starting ollama server..."
            nohup ollama serve > /tmp/ollama.log 2>&1 &
        fi
    fi
}

# =============================================================================
# Python Dependencies Installation
# =============================================================================

install_python_deps() {
    log_info "Getting requirements..."
    pip install --upgrade pip
    
    log_info "Installing GenRL..."
    pip install -r code_gen_exp/requirements.txt
}

# =============================================================================
# Configuration Management
# =============================================================================

setup_config() {
    log_info "Setting up configuration..."
    
    if [[ ! -d "$CONFIG_DIR" ]]; then
        mkdir "$CONFIG_DIR"
    fi
    
    local config_file="$CONFIG_DIR/code-gen-swarm.yaml"
    local default_config="$ROOT/code_gen_exp/config/code-gen-swarm.yaml"
    
    if [[ -f "$config_file" ]]; then
        if ! cmp -s "$default_config" "$config_file"; then
            if [[ -z "$GENSYN_RESET_CONFIG" ]]; then
                log_warn "Found differences in code-gen-swarm.yaml. Set GENSYN_RESET_CONFIG to reset to default."
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
    yn=${yn:-N}
    
    case $yn in
        [Yy]*) 
            read -p "Enter your Hugging Face access token: " HUGGINGFACE_ACCESS_TOKEN 
            ;;
        [Nn]*) 
            HUGGINGFACE_ACCESS_TOKEN="None" 
            ;;
        *) 
            echo ">>> No answer was given, so NO models will be pushed to Hugging Face Hub"
            HUGGINGFACE_ACCESS_TOKEN="None" 
            ;;
    esac

    echo -en $GREEN_TEXT
    read -p ">> Enter the name of the model you want to use in huggingface repo/name format, or press [Enter] to use the default model. " MODEL_NAME
    echo -en $RESET_TEXT

    if [[ -n "$MODEL_NAME" ]]; then
        export MODEL_NAME
        echo_green ">> Using model: $MODEL_NAME"
    else
        echo_green ">> Using default model from config"
    fi
    
    # Logout from HuggingFace to prevent env issues
    if ! hf auth logout > /dev/null 2>&1; then
        unset HF_TOKEN
        unset HUGGING_FACE_HUB_TOKEN
        hf auth logout > /dev/null 2>&1 || true
    fi
}

# =============================================================================
# Enhanced Error Handling and Process Management
# =============================================================================

# Cleanup server process
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        log_info "Stopping modal login server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
    fi
}

# Handle interruption signals
handle_interrupt() {
    log_warn "Received interrupt signal (Ctrl+C)..."
    cleanup
}

cleanup() {
    log_info "Shutting down trainer..."
    cleanup_server
    # Kill all processes belonging to this script's process group
    kill -- -$$ 2>/dev/null || true
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

    From Gensyn

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
        
        install_nodejs
        install_yarn
        setup_modal_login
        wait_for_user_data
        wait_for_api_activation
    fi
    
    # Install dependencies
    install_python_deps
    install_ollama
    setup_config
    
    echo_green ">> Done!"
    
    # Get user preferences
    get_user_preferences
    
    # Final messages
    echo -en $RESET_TEXT
    echo_green ">> Good luck in the swarm!"
    echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"
    
    # Launch the swarm
    log_info "Starting swarm launcher..."
    python -m code_gen_exp.runner.swarm_launcher \
        --config-path "$ROOT/code_gen_exp/config" \
        --config-name "code-gen-swarm.yaml"
    
    wait  # Keep script running until Ctrl+C
}

# Execute main function
main "$@"
