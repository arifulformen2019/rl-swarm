import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend, TrainingPhase, TrainingStateManager
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami

from rgym_exp.src.utils.name_utils import get_name_from_peer_id
from rgym_exp.src.prg_module import PRGModule

# Enhanced colorful logging with emoji support
try:
    from colorama import Fore, Style, Back, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class MockColor:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = WHITE = LIGHTGREEN_EX = LIGHTRED_EX = LIGHTBLUE_EX = ""
    class MockStyle:
        RESET_ALL = BRIGHT = DIM = ""
    class MockBack:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
    Fore = MockColor()
    Style = MockStyle()
    Back = MockBack()


class BeautifulLogger:
    """Enhanced logger with beautiful formatting and emoji support"""
    
    EMOJIS = {
        'start': '🚀',
        'success': '✅', 
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'rocket': '🚀',
        'fire': '🔥',
        'crown': '👑',
        'gem': '💎',
        'lightning': '⚡',
        'shield': '🛡️',
        'brain': '🧠',
        'robot': '🤖',
        'chart': '📊',
        'trophy': '🏆',
        'gear': '⚙️',
        'sync': '🔄',
        'heart': '💓',
        'star': '⭐',
        'celebration': '🎉',
        'hourglass': '⏳',
        'clock': '🕐',
        'save': '💾',
        'upload': '📤',
        'download': '📥',
        'network': '🌐',
        'chain': '⛓️',
    }
    
    @classmethod
    def create_box_message(cls, title: str, content: list, width: int = 70, emoji: str = ''):
        """Create beautiful box-style message"""
        if not COLORAMA_AVAILABLE:
            result = f"\n=== {title} ===\n"
            for line in content:
                result += f"  {line}\n"
            result += "=" * (len(title) + 8) + "\n"
            return result
            
        # Header
        header = f"{emoji} {title}" if emoji else title
        
        lines = [
            f"{Fore.CYAN}{'═' * width}",
            f"{Fore.CYAN}║{Style.BRIGHT}{Fore.WHITE} {header:<{width-4}} {Fore.CYAN}║",
            f"{Fore.CYAN}╠{'═' * (width-2)}╣"
        ]
        
        # Content lines
        for line in content:
            if len(line) > width - 6:
                # Word wrap for long lines
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) > width - 8:
                        lines.append(f"{Fore.CYAN}║ {Fore.WHITE}{current_line:<{width-4}} {Fore.CYAN}║")
                        current_line = word + " "
                    else:
                        current_line += word + " "
                if current_line.strip():
                    lines.append(f"{Fore.CYAN}║ {Fore.WHITE}{current_line.strip():<{width-4}} {Fore.CYAN}║")
            else:
                lines.append(f"{Fore.CYAN}║ {Fore.WHITE}{line:<{width-4}} {Fore.CYAN}║")
        
        # Footer
        lines.append(f"{Fore.CYAN}{'═' * width}{Style.RESET_ALL}")
        
        for line in lines:
            print(line)
        
        return ""
    
    @classmethod
    def status_line(cls, label: str, value: str, status: str = "info", width: int = 50):
        """Create beautiful status line"""
        if not COLORAMA_AVAILABLE:
            return f"{label}: {value}"
        
        colors = {
            "success": Fore.LIGHTGREEN_EX,
            "error": Fore.LIGHTRED_EX,
            "warning": Fore.YELLOW,
            "info": Fore.LIGHTBLUE_EX,
            "neutral": Fore.WHITE
        }
        
        emoji = {
            "success": cls.EMOJIS['success'],
            "error": cls.EMOJIS['error'], 
            "warning": cls.EMOJIS['warning'],
            "info": cls.EMOJIS['info'],
            "neutral": ""
        }
        
        color = colors.get(status, Fore.WHITE)
        icon = emoji.get(status, "")
        
        # Format: [ICON] Label: Value
        formatted = f"{icon} {Style.BRIGHT}{label}:{Style.RESET_ALL} {color}{value}{Style.RESET_ALL}"
        return formatted
    
    @classmethod
    def progress_bar(cls, current: int, total: int, width: int = 30, label: str = ""):
        """Create beautiful progress bar"""
        if not COLORAMA_AVAILABLE:
            return f"{label} {current}/{total}"
            
        percentage = (current / total) if total > 0 else 0
        filled = int(width * percentage)
        
        bar = (f"{Back.GREEN}{' ' * filled}"
               f"{Back.WHITE}{' ' * (width - filled)}"
               f"{Style.RESET_ALL}")
        
        return f"{label} {bar} {percentage:.1%} ({current}/{total})"
    
    @classmethod
    def section_header(cls, title: str, emoji: str = "", level: int = 1):
        """Create beautiful section header"""
        if not COLORAMA_AVAILABLE:
            return f"=== {title} ==="
            
        symbols = ["═", "─", "·"]
        symbol = symbols[min(level-1, len(symbols)-1)]
        
        header_emoji = emoji if emoji else cls.EMOJIS.get('star', '')
        full_title = f"{header_emoji} {title}"
        
        width = max(60, len(full_title) + 10)
        line = symbol * width
        
        return f"\n{Fore.MAGENTA}{Style.BRIGHT}{line}\n{full_title:^{width}}\n{line}{Style.RESET_ALL}\n"


class CrashSafeSwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """Enhanced SwarmGameManager with crash protection and beautiful logging."""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        # Crash protection parameters
        enable_crash_protection: bool = True,
        enable_dht_auto_restart: bool = True,
        memory_threshold_mb: int = 1800,
        restart_interval_minutes: int = 30,
        max_auto_restarts: int = 15,
        health_check_interval: int = 60,
        **kwargs,
    ):
        # Set crash protection attributes FIRST
        self.enable_crash_protection = enable_crash_protection
        self.memory_threshold_mb = float(memory_threshold_mb)
        self.restart_interval_minutes = float(restart_interval_minutes)
        self.max_auto_restarts = int(max_auto_restarts)
        self.health_check_interval = health_check_interval
        self._last_health_log_time = 0
        self.coordinator = coordinator
        
        # Initialize beautiful logger
        self.logger = BeautifulLogger()
        
        # Call parent constructor
        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # Initialize crash protection system
        if self.enable_crash_protection:
            self._init_crash_protection()
        else:
            self.training_state_manager = None
            get_logger().warning(self.logger.status_line(
                "Crash Protection", "DISABLED", "warning"
            ))

        # Setup peer identity and model info
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        
        # Model name handling
        model_name = self._get_model_name()
        self.model_display_name = self._clean_model_name(model_name)
        
        # Setup logging
        self._setup_logging(log_dir, model_name)

        # Initialize blockchain components
        self._init_blockchain_components()

        # Initialize training components
        self._init_training_components(log_dir, hf_token, hf_push_frequency, **kwargs)

        # Log final initialization status
        self._log_initialization_status()

    def _init_crash_protection(self):
        """Initialize crash protection system with beautiful logging"""
        self.training_state_manager = TrainingStateManager()
        
        # Register with DHT backend
        self.communication.register_training_state_manager(self.training_state_manager)
        self.communication.set_restart_callback(self._on_dht_restart_event)
        
        # Ensure backend has correct parameter types
        if hasattr(self.communication, 'memory_threshold_mb'):
            self.communication.memory_threshold_mb = self.memory_threshold_mb
        if hasattr(self.communication, 'restart_interval_minutes'):
            self.communication.restart_interval_minutes = self.restart_interval_minutes
        if hasattr(self.communication, 'max_auto_restarts'):
            self.communication.max_auto_restarts = self.max_auto_restarts
            
        get_logger().info(self.logger.status_line(
            "Crash Protection", "INITIALIZED", "success"
        ))

    def _get_model_name(self) -> str:
        """Get model name from trainer"""
        if hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm:
            return getattr(self.trainer, "model_name", "vLLM_Model")
        
        config_obj = getattr(getattr(self.trainer, "model", None), "config", None)
        if config_obj:
            return getattr(config_obj, "_name_or_path", "UnknownModel")
        return "UnknownModel"

    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name for display"""
        clean_name = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Remove common suffixes
        for suffix in ["-Instruct", "-Chat", "-Base", "-v1", "-v2", "-v3"]:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        return clean_name

    def _setup_logging(self, log_dir: str, model_name: str):
        """Setup beautiful logging configuration"""
        # Enhanced format with colors and emoji
        if COLORAMA_AVAILABLE:
            format_msg = f"{Fore.CYAN}[{Style.BRIGHT}{{model}}{Style.RESET_ALL}{Fore.CYAN}] {Fore.WHITE}%(asctime)s {Fore.YELLOW}%(levelname)s{Fore.WHITE}: %(message)s{Style.RESET_ALL}"
        else:
            format_msg = f"[{self.model_display_name}] %(asctime)s %(levelname)s: %(message)s"
        
        # Capture model_display_name in closure
        model_display_name = self.model_display_name
        
        # Custom formatter
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                if COLORAMA_AVAILABLE:
                    # Color by log level
                    level_colors = {
                        'DEBUG': Fore.BLUE,
                        'INFO': Fore.WHITE,
                        'WARNING': Fore.YELLOW,
                        'ERROR': Fore.RED,
                        'CRITICAL': Fore.MAGENTA + Style.BRIGHT
                    }
                    
                    original_format = self._style._fmt
                    # Sử dụng biến từ closure thay vì self.model_display_name
                    self._style._fmt = original_format.replace('{model}', model_display_name)
                    
                    # Format with colors
                    formatted = super().format(record)
                    self._style._fmt = original_format
                    return formatted
                else:
                    return super().format(record)
        
        logging.basicConfig(level=logging.INFO, format=format_msg.replace('{model}', self.model_display_name))
        
        # File handler with custom formatter
        formatter = ColoredFormatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        get_logger().addHandler(file_handler)
        
        get_logger().info(self.logger.status_line(
            "Model Loaded", model_name, "success"
        ))


    def _init_blockchain_components(self):
        """Initialize blockchain-related components with beautiful logging"""
        get_logger().info(self.logger.section_header(
            "Blockchain Initialization", self.logger.EMOJIS['chain'], 2
        ))
        
        self.coordinator.register_peer(self.peer_id)
        
        # Get current round and sync communication
        round_num, _ = self.coordinator.get_round_and_stage()
        self.state.round = round_num
        self.communication.step_ = self.state.round
        
        # Setup submission tracking
        self.batched_signals = 0.0
        self.time_since_submit = time.time()
        self.submit_period = 2.0  # hours
        self.submitted_this_round = False
        self.round_counter = 0
        
        get_logger().info(self.logger.status_line(
            "Blockchain Sync", f"Round {round_num}", "success"
        ))

    def _init_training_components(self, log_dir: str, hf_token: str | None, hf_push_frequency: int, **kwargs):
        """Initialize training-related components with beautiful logging"""
        get_logger().info(self.logger.section_header(
            "Training Components Setup", self.logger.EMOJIS['brain'], 2
        ))
        
        # Setup Hugging Face integration
        self.hf_token = hf_token
        self.hf_push_frequency = hf_push_frequency
        self._setup_huggingface_integration()
        
        # Initialize PRG Game
        self.prg_module = PRGModule(log_dir, **kwargs)
        self.prg_game = self.prg_module.prg_game
        
        get_logger().info(self.logger.status_line(
            "PRG Module", "LOADED", "success"
        ))
        
        # Write system info
        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

    def _setup_huggingface_integration(self):
        """Setup Hugging Face Hub integration with beautiful logging"""
        if (self.hf_token not in [None, "None"] and 
            not (hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm)):
            try:
                username = whoami(token=self.hf_token)["name"]
                model_name_suffix = self._get_model_name().split("/")[-1]
                hub_model_id = f"{username}/{model_name_suffix}-Gensyn-Swarm-{self.animal_name}"
                
                self.trainer.args.hub_model_id = hub_model_id
                self.trainer.args.push_to_hub = True
                self.trainer.args.hub_token = self.hf_token
                
                get_logger().info(self.logger.status_line(
                    "Hugging Face Hub", f"Connected as {username}", "success"
                ))
                login(self.hf_token)
            except Exception as e:
                get_logger().warning(self.logger.status_line(
                    "Hugging Face Setup", f"Failed: {e}", "error"
                ))
        else:
            get_logger().info(self.logger.status_line(
                "Hugging Face Hub", "DISABLED", "neutral"
            ))

    def _log_initialization_status(self):
        """Log final initialization status with beautiful design"""
        protection_status = "ENABLED" if self.enable_crash_protection else "DISABLED"
        protection_emoji = self.logger.EMOJIS['shield'] if self.enable_crash_protection else self.logger.EMOJIS['warning']
        
        content = [
            f"Model: {self.model_display_name}",
            f"Agent: {self.animal_name} {self.logger.EMOJIS['robot']}",
            f"Peer ID: {self.peer_id[:16]}...",
            f"Starting Round: {self.state.round} {self.logger.EMOJIS['rocket']}",
            f"Crash Protection: {protection_status} {protection_emoji}"
        ]
        
        box_message = self.logger.create_box_message(
            "SWARM MANAGER INITIALIZED", content, 
            emoji=self.logger.EMOJIS['celebration']
        )
        
        get_logger().info(box_message)

    def _on_dht_restart_event(self, event_type: str, reason: str):
        """Handle DHT restart events with beautiful logging"""
        if event_type == "restart_completed":
            get_logger().info(self.logger.status_line(
                "DHT Restart", f"COMPLETED - {reason}", "success"
            ))
        elif event_type == "restart_failed":
            get_logger().error(self.logger.status_line(
                "DHT Restart", f"FAILED - {reason}", "error"
            ))

    def _execute_pending_restart(self):
        """Execute pending DHT restart with beautiful logging"""
        if not (self.training_state_manager and self.training_state_manager._restart_requested):
            return
            
        reason = self.training_state_manager._restart_reason
        
        get_logger().info(self.logger.section_header(
            f"Executing DHT Restart: {reason}", self.logger.EMOJIS['sync'], 2
        ))
        
        try:
            if hasattr(self.communication, 'perform_coordinated_restart'):
                self.communication.perform_coordinated_restart(reason)
                self.training_state_manager.acknowledge_restart()
                
                get_logger().info(self.logger.status_line(
                    "DHT Restart", "COMPLETED SUCCESSFULLY", "success"
                ))
            else:
                get_logger().error(self.logger.status_line(
                    "DHT Restart", "METHOD NOT FOUND", "error"
                ))
                self.training_state_manager.acknowledge_restart()
                
        except Exception as e:
            get_logger().error(self.logger.status_line(
                "DHT Restart", f"EXECUTION FAILED: {e}", "error"
            ))
            self.training_state_manager.acknowledge_restart()

    def _safe_blockchain_submit(self, signal_by_agent):
        """Thread-safe blockchain submit with beautiful logging"""
        if not self.enable_crash_protection or not self.training_state_manager:
            return self._try_submit_to_chain(signal_by_agent)
        
        # Enter critical section
        self.training_state_manager.enter_critical_section("blockchain_submit")
        
        try:
            return self._try_submit_to_chain(signal_by_agent)
        except Exception as e:
            get_logger().error(self.logger.status_line(
                "Blockchain Submit", f"FAILED: {e}", "error"
            ))
            raise
        finally:
            # Always exit critical section
            self.training_state_manager.exit_critical_section("blockchain_submit")

    def _get_total_rewards_by_agent(self):
        """Calculate total rewards by agent"""
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    total = sum(sum(generation_rewards) for generation_rewards in batch_rewards)
                    rewards_by_agent[agent_id] += total
        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        """Get rewards for this agent"""
        if not signal_by_agent:
            return 0
        
        my_signal = signal_by_agent.get(self.peer_id, 0)
        return (my_signal + 1) * (my_signal > 0) + my_signal * (my_signal <= 0)

    def _try_submit_to_chain(self, signal_by_agent):
        """Submit results to blockchain with beautiful logging"""
        elapsed_hours = (time.time() - self.time_since_submit) / 3600
        
        if elapsed_hours > self.submit_period:
            try:
                get_logger().info(self.logger.section_header(
                    f"Blockchain Submission - Round {self.state.round}", 
                    self.logger.EMOJIS['chain'], 3
                ))
                
                points = int(self.batched_signals)
                get_logger().info(self.logger.status_line(
                    "Submitting Points", f"{points} points", "info"
                ))
                
                # Submit reward and winners
                self.coordinator.submit_reward(self.state.round, 0, points, self.peer_id)
                
                if signal_by_agent:
                    max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                    winner_name = get_name_from_peer_id(max_agent, True)
                    get_logger().info(self.logger.status_line(
                        "Round Winner", f"{winner_name} ({max_signal} points) {self.logger.EMOJIS['crown']}", "success"
                    ))
                else:
                    max_agent, max_signal = self.peer_id, points
                
                self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)
                
                # Reset counters
                self.batched_signals = 0.0
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                
                get_logger().info(self.logger.status_line(
                    "Blockchain Submission", "SUCCESS", "success"
                ))
                
            except Exception as e:
                get_logger().error(self.logger.status_line(
                    "Blockchain Submission", f"FAILED: {e}", "error"
                ))
                get_logger().exception("Full blockchain submission error")
        else:
            remaining_minutes = (self.submit_period - elapsed_hours) * 60
            
            # Log every 30 minutes when waiting
            if not hasattr(self, '_last_waiting_log'):
                self._last_waiting_log = 0
            
            if time.time() - self._last_waiting_log > 1200:  # 30 minutes
                get_logger().info(self.logger.status_line(
                    "Next Blockchain Submit", f"in {remaining_minutes:.0f} minutes {self.logger.EMOJIS['hourglass']}", "info"
                ))
                self._last_waiting_log = time.time()

    def _hook_after_rewards_updated(self):
        """Handle reward updates with beautiful logging"""
        # Set training phase
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.MODEL_UPDATE)
        
        signal_by_agent = self._get_total_rewards_by_agent()
        old_signals = self.batched_signals
        self.batched_signals += self._get_my_rewards(signal_by_agent)
        
        # Log reward updates with beautiful format
        reward_gained = self.batched_signals - old_signals
        if reward_gained > 0:
            get_logger().info(self.logger.status_line(
                "Reward Gained", f"+{reward_gained:.1f} points {self.logger.EMOJIS['gem']} (Total: {int(self.batched_signals)})", "success"
            ))
        
        # Submit to blockchain
        self._safe_blockchain_submit(signal_by_agent)
        
        # Reset to idle phase
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.IDLE)

    def _hook_after_round_advanced(self):
        """Handle round advancement with beautiful logging"""
        self.round_counter += 1
        
        get_logger().info(self.logger.section_header(
            f"NEW ROUND STARTED: {self.state.round}", self.logger.EMOJIS['rocket'], 1
        ))
        
        get_logger().info(self.logger.status_line(
            "Total Rounds Completed", f"{self.round_counter} rounds {self.logger.EMOJIS['trophy']}", "info"
        ))
        
        # Log system health periodically
        self._log_system_health()
        
        # PRG Game logic with enhanced logging
        if self.prg_game:
            if self.training_state_manager:
                self.training_state_manager.set_phase(TrainingPhase.PRG_GAME)
            
            get_logger().info(self.logger.section_header(
                "PRG Game Logic", self.logger.EMOJIS['brain'], 2
            ))
            
            try:
                prg_history_dict = self.prg_module.prg_history_dict
                results_dict = self.trainer.play_prg_game_logits(prg_history_dict)
                self.prg_module.play_prg_game(results_dict, self.peer_id)
                
                get_logger().info(self.logger.status_line(
                    "PRG Game", "COMPLETED SUCCESSFULLY", "success"
                ))
            except Exception as e:
                get_logger().error(self.logger.status_line(
                    "PRG Game", f"FAILED: {e}", "error"
                ))
                get_logger().exception("PRG Game error details")
            finally:
                if self.training_state_manager:
                    self.training_state_manager.set_phase(TrainingPhase.IDLE)
        
        # Save model to Hugging Face
        self._save_to_hf()

        # Final blockchain submit if needed
        if not self.submitted_this_round:
            signal_by_agent = self._get_total_rewards_by_agent()
            self._safe_blockchain_submit(signal_by_agent)
        
        self.submitted_this_round = False

        # Block until next round - THIS IS WHERE RESTART HAPPENS
        self.agent_block()

    def _hook_after_game(self):
        """Handle game completion with beautiful logging"""
        get_logger().info(self.logger.section_header(
            "GAME COMPLETION - Final Cleanup", self.logger.EMOJIS['celebration'], 1
        ))
        
        # Log final health status
        if self.enable_crash_protection:
            self._log_comprehensive_health_status()
        
        # Final model save
        self._save_to_hf()
        
        # Clean shutdown
        if hasattr(self.communication, 'shutdown'):
            self.communication.shutdown()
            get_logger().info(self.logger.status_line(
                "System Shutdown", "COMPLETED", "success"
            ))

    def _save_to_hf(self):
        """Save model to Hugging Face Hub with beautiful logging"""
        if (self.hf_token not in [None, "None"] and 
            self.state.round % self.hf_push_frequency == 0):
            
            get_logger().info(self.logger.section_header(
                f"Hugging Face Upload - Round {self.state.round}", self.logger.EMOJIS['upload'], 3
            ))
            
            try:
                repo_id = getattr(self.trainer.args, 'hub_model_id', None) or Path(self.trainer.args.output_dir).name
                
                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=["rl-swarm", "genrl-swarm", "grpo", "gensyn", f"I am {self.animal_name}"]
                )
                
                get_logger().info(self.logger.status_line(
                    "Model Upload", f"SUCCESS to {repo_id} {self.logger.EMOJIS['success']}", "success"
                ))
                
            except Exception as e:
                get_logger().error(self.logger.status_line(
                    "Model Upload", f"FAILED: {e}", "error"
                ))

    def agent_block(self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15):
        """Enhanced agent blocking with beautiful logging and centralized restart execution"""
        
        # Set idle phase - this is the safest time for DHT restarts
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.IDLE)
        
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        
        get_logger().info(self.logger.section_header(
            f"Waiting for Swarm Round Advancement", self.logger.EMOJIS['hourglass'], 2
        ))
        
        get_logger().info(self.logger.status_line(
            "Agent Status", f"Blocking for next round ({self.animal_name})", "info"
        ))
        
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            
            # CENTRALIZED RESTART EXECUTION - This is the ONLY place restarts happen
            self._execute_pending_restart()
            
            # Perform basic DHT health check (non-critical)
            try:
                if self.communication.dht:
                    _ = self.communication.dht.get_visible_maddrs(latest=True)
            except Exception as e:
                get_logger().debug(f"DHT health check during blocking: {e}")

            # Check for round advancement
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(self.logger.status_line(
                        "Round Fetch", f"Failed: {e}", "warning"
                    ))
                    fetch_log_time = curr_time
                time.sleep(check_interval)
                continue

            # Check if we can advance to next round
            if round_num >= self.state.round:
                get_logger().info(self.logger.status_line(
                    "Round Advancement", f"Joining round {round_num} {self.logger.EMOJIS['rocket']}", "success"
                ))
                check_backoff = check_interval  # Reset backoff
                self.state.round = round_num
                return
            else:
                get_logger().info(self.logger.status_line(
                    "Round Status", f"Already finished round {round_num}. Next check in {check_backoff}s", "info"
                ))
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            # Check for final round
            if round_num == self.max_round - 1:
                get_logger().info(self.logger.status_line(
                    "Training Complete", f"Reached maximum round: {self.max_round} {self.logger.EMOJIS['celebration']}", "success"
                ))
                return

        get_logger().info(self.logger.status_line(
            "Training Timeout", f"After {self.train_timeout}s {self.logger.EMOJIS['clock']}", "warning"
        ))

    def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        status = {
            "manager_info": {
                "peer_id": self.peer_id,
                "animal_name": self.animal_name,
                "round": self.state.round,
                "batched_signals": self.batched_signals,
                "round_counter": self.round_counter,
                "crash_protection_enabled": self.enable_crash_protection,
            },
        }
        
        if self.training_state_manager:
            status["training_state"] = self.training_state_manager.get_stats()
        
        if hasattr(self.communication, 'get_auto_restart_status'):
            status["dht_auto_restart"] = self.communication.get_auto_restart_status()
            
        if hasattr(self.communication, 'get_status'):
            status["dht_backend"] = self.communication.get_status()
            
        return status

    def _log_system_health(self):
        """Log system health status periodically with beautiful format"""
        current_time = time.time()
        
        if current_time - self._last_health_log_time < self.health_check_interval:
            return
            
        self._last_health_log_time = current_time
        
        if not self.enable_crash_protection:
            return
            
        status = self.get_comprehensive_health_status()
        
        # Extract key metrics
        training_phase = status.get("training_state", {}).get("current_phase", "unknown")
        restart_count = status.get("dht_auto_restart", {}).get("restart_count", 0)
        emergency_mode = status.get("dht_backend", {}).get("emergency_mode", False)
        dht_mode = status.get("dht_backend", {}).get("mode", "unknown")
        
        # Create beautiful health status
        health_content = [
            f"Phase: {training_phase} {self.logger.EMOJIS['gear']}",
            f"Mode: {dht_mode} {self.logger.EMOJIS['network']}",
            f"Restarts: {restart_count} {self.logger.EMOJIS['sync']}"
        ]
        
        if emergency_mode:
            health_content.append(f"Emergency Mode: ACTIVE {self.logger.EMOJIS['warning']}")
        
        box_message = self.logger.create_box_message(
            "HEALTH STATUS", health_content, 
            emoji=self.logger.EMOJIS['heart']
        )
        
        get_logger().info(box_message)
        
        if restart_count > 5:
            get_logger().warning(self.logger.status_line(
                "High Restart Count", f"{restart_count} restarts detected", "warning"
            ))

    def _log_comprehensive_health_status(self):
        """Log detailed health status with beautiful format"""
        if not self.enable_crash_protection:
            return
            
        status = self.get_comprehensive_health_status()
        
        get_logger().info(self.logger.section_header(
            "COMPREHENSIVE HEALTH STATUS", self.logger.EMOJIS['shield'], 1
        ))
        
        # Manager info
        manager_info = status.get("manager_info", {})
        manager_content = [
            f"Agent: {manager_info.get('animal_name')} {self.logger.EMOJIS['robot']}",
            f"Round: {manager_info.get('round')} {self.logger.EMOJIS['trophy']}",
            f"Points: {manager_info.get('batched_signals')} {self.logger.EMOJIS['gem']}",
            f"Total Rounds: {manager_info.get('round_counter', 0)} {self.logger.EMOJIS['chart']}"
        ]
        
        manager_box = self.logger.create_box_message(
            "AGENT STATUS", manager_content, 
            emoji=self.logger.EMOJIS['robot']
        )
        get_logger().info(manager_box)
        
        # Training state
        training_state = status.get("training_state", {})
        if training_state:
            training_content = [
                f"Current Phase: {training_state.get('current_phase')} {self.logger.EMOJIS['gear']}",
                f"Total Restarts: {training_state.get('total_restarts', 0)} {self.logger.EMOJIS['sync']}",
                f"Critical Sections: {training_state.get('critical_sections', 0)} {self.logger.EMOJIS['shield']}"
            ]
            
            training_box = self.logger.create_box_message(
                "TRAINING STATE", training_content,
                emoji=self.logger.EMOJIS['brain']
            )
            get_logger().info(training_box)
        
        # DHT status
        dht_backend = status.get("dht_backend", {})
        if dht_backend:
            dht_content = [
                f"DHT Mode: {dht_backend.get('mode')} {self.logger.EMOJIS['network']}",
                f"Emergency Mode: {dht_backend.get('emergency_mode')} {self.logger.EMOJIS['warning'] if dht_backend.get('emergency_mode') else self.logger.EMOJIS['success']}",
                f"Connection Status: {dht_backend.get('connection_status', 'unknown')} {self.logger.EMOJIS['chain']}"
            ]
            
            dht_box = self.logger.create_box_message(
                "DHT BACKEND STATUS", dht_content,
                emoji=self.logger.EMOJIS['network']
            )
            get_logger().info(dht_box)


# Backward compatibility alias
SwarmGameManager = CrashSafeSwarmGameManager


# Factory function for enhanced manager creation
def create_crash_safe_swarm_manager(**kwargs):
    """Create SwarmGameManager with crash protection and beautiful logging"""
    
    # Set default crash protection parameters
    crash_defaults = {
        'enable_crash_protection': True,
        'enable_dht_auto_restart': True,
        'memory_threshold_mb': 1800,
        'restart_interval_minutes': 30,
        'max_auto_restarts': 15,
        'health_check_interval': 60,
    }
    
    # Merge defaults with user parameters
    for key, default_value in crash_defaults.items():
        kwargs.setdefault(key, default_value)
    
    return CrashSafeSwarmGameManager(**kwargs)


# Emergency control functions with beautiful logging
def emergency_disable_crash_protection(manager):
    """Emergency function to disable crash protection with beautiful logging"""
    if hasattr(manager, 'enable_crash_protection'):
        manager.enable_crash_protection = False
        
        if hasattr(manager, 'logger'):
            get_logger().warning(manager.logger.status_line(
                "CRASH PROTECTION", "EMERGENCY DISABLED", "error"
            ))
        else:
            get_logger().warning("CRASH PROTECTION EMERGENCY DISABLED")
        
        if hasattr(manager.communication, 'auto_restart_enabled'):
            manager.communication.auto_restart_enabled = False


def get_system_health_report(manager) -> str:
    """Get formatted system health report with beautiful format"""
    if not hasattr(manager, 'get_comprehensive_health_status'):
        return "Health monitoring not available"
        
    status = manager.get_comprehensive_health_status()
    
    if hasattr(manager, 'logger'):
        logger = manager.logger
        
        # Create beautiful health report
        content = []
        
        # Manager status
        manager_info = status.get("manager_info", {})
        content.append(f"Agent: {manager_info.get('animal_name', 'unknown')} {logger.EMOJIS['robot']}")
        content.append(f"Round: {manager_info.get('round', 0)} {logger.EMOJIS['trophy']}")
        content.append(f"Points: {manager_info.get('batched_signals', 0)} {logger.EMOJIS['gem']}")
        
        # Training state
        training_state = status.get("training_state", {})
        if training_state:
            content.append(f"Phase: {training_state.get('current_phase', 'unknown')} {logger.EMOJIS['gear']}")
            content.append(f"Restarts: {training_state.get('total_restarts', 0)} {logger.EMOJIS['sync']}")
        
        # DHT status
        dht_backend = status.get("dht_backend", {})
        if dht_backend:
            content.append(f"DHT Mode: {dht_backend.get('mode', 'unknown')} {logger.EMOJIS['network']}")
            
            if dht_backend.get('emergency_mode', False):
                content.append(f"DHT Emergency Mode: ACTIVE {logger.EMOJIS['warning']}")
        
        return logger.create_box_message("SYSTEM HEALTH REPORT", content, emoji=logger.EMOJIS['heart'])
    else:
        # Fallback to simple format
        report = ["=== SYSTEM HEALTH REPORT ==="]
        
        # Manager status
        manager_info = status.get("manager_info", {})
        report.append(f"Agent: {manager_info.get('animal_name', 'unknown')}")
        report.append(f"Round: {manager_info.get('round', 0)}")
        report.append(f"Points: {manager_info.get('batched_signals', 0)}")
        
        # Training state
        training_state = status.get("training_state", {})
        if training_state:
            report.append(f"Phase: {training_state.get('current_phase', 'unknown')}")
            report.append(f"Restarts: {training_state.get('total_restarts', 0)}")
        
        # DHT status
        dht_backend = status.get("dht_backend", {})
        if dht_backend:
            report.append(f"DHT Mode: {dht_backend.get('mode', 'unknown')}")
            
            if dht_backend.get('emergency_mode', False):
                report.append("DHT Emergency Mode: ACTIVE")
        
        report.append("========================")
        
        return "\n".join(report)


# Additional utility functions for beautiful logging
def log_performance_metrics(manager, metrics: Dict[str, Any]):
    """Log performance metrics with beautiful format"""
    if not hasattr(manager, 'logger'):
        get_logger().info(f"Performance Metrics: {metrics}")
        return
    
    logger = manager.logger
    
    content = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key.lower() in ['accuracy', 'score', 'reward']:
                emoji = logger.EMOJIS['trophy']
            elif key.lower() in ['time', 'duration', 'latency']:
                emoji = logger.EMOJIS['clock']
            elif key.lower() in ['memory', 'cpu', 'gpu']:
                emoji = logger.EMOJIS['gear']
            else:
                emoji = logger.EMOJIS['chart']
            
            content.append(f"{key}: {value} {emoji}")
        else:
            content.append(f"{key}: {value}")
    
    box_message = logger.create_box_message(
        "PERFORMANCE METRICS", content,
        emoji=logger.EMOJIS['chart']
    )
    
    get_logger().info(box_message)


def log_training_progress(manager, current_step: int, total_steps: int, loss: float = None):
    """Log training progress with beautiful progress bar"""
    if not hasattr(manager, 'logger'):
        progress = f"Training Progress: {current_step}/{total_steps}"
        if loss is not None:
            progress += f" (Loss: {loss:.4f})"
        get_logger().info(progress)
        return
    
    logger = manager.logger
    
    # Create progress bar
    progress_bar = logger.progress_bar(
        current_step, total_steps, 
        label=f"{logger.EMOJIS['rocket']} Training Progress"
    )
    
    content = [progress_bar]
    
    if loss is not None:
        content.append(f"Current Loss: {loss:.4f} {logger.EMOJIS['chart']}")
    
    content.append(f"Agent: {manager.animal_name} {logger.EMOJIS['robot']}")
    
    box_message = logger.create_box_message(
        "TRAINING PROGRESS", content,
        emoji=logger.EMOJIS['brain']
    )
    
    get_logger().info(box_message)
