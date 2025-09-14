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


class CrashSafeSwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """Enhanced SwarmGameManager with crash protection and clean logging."""

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
            get_logger().warning(f"{Fore.YELLOW}⚠️ [SWARM MANAGER] Crash Protection: DISABLED{Style.RESET_ALL}")

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
        """Initialize crash protection system with clean logging"""
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
            
        get_logger().info(f"{Fore.GREEN}✅ [SWARM MANAGER] Crash Protection: INITIALIZED{Style.RESET_ALL}")

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
        """Setup clean logging configuration"""
        # Enhanced format with colors
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
                    original_format = self._style._fmt
                    # Use variable from closure instead of self.model_display_name
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
        
        get_logger().info(f"{Fore.GREEN}✅ [SWARM MANAGER] Model Loaded: {model_name}{Style.RESET_ALL}")

    def _init_blockchain_components(self):
        """Initialize blockchain-related components with clean logging"""
        get_logger().info(f"{Fore.CYAN}⛓️ [BLOCKCHAIN] Initializing components...{Style.RESET_ALL}")
        
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
        
        get_logger().info(f"{Fore.GREEN}✅ [BLOCKCHAIN] Synced to round {round_num}{Style.RESET_ALL}")

    def _init_training_components(self, log_dir: str, hf_token: str | None, hf_push_frequency: int, **kwargs):
        """Initialize training-related components with clean logging"""
        get_logger().info(f"{Fore.CYAN}🧠 [TRAINING] Setting up components...{Style.RESET_ALL}")
        
        # Setup Hugging Face integration
        self.hf_token = hf_token
        self.hf_push_frequency = hf_push_frequency
        self._setup_huggingface_integration()
        
        # Initialize PRG Game
        self.prg_module = PRGModule(log_dir, **kwargs)
        self.prg_game = self.prg_module.prg_game
        
        get_logger().info(f"{Fore.GREEN}✅ [TRAINING] PRG Module loaded{Style.RESET_ALL}")
        
        # Write system info
        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

    def _setup_huggingface_integration(self):
        """Setup Hugging Face Hub integration with clean logging"""
        if (self.hf_token not in [None, "None"] and 
            not (hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm)):
            try:
                username = whoami(token=self.hf_token)["name"]
                model_name_suffix = self._get_model_name().split("/")[-1]
                hub_model_id = f"{username}/{model_name_suffix}-Gensyn-Swarm-{self.animal_name}"
                
                self.trainer.args.hub_model_id = hub_model_id
                self.trainer.args.push_to_hub = True
                self.trainer.args.hub_token = self.hf_token
                
                get_logger().info(f"{Fore.GREEN}✅ [HUGGING FACE] Connected as {username}{Style.RESET_ALL}")
                login(self.hf_token)
            except Exception as e:
                get_logger().warning(f"{Fore.YELLOW}⚠️ [HUGGING FACE] Setup failed: {e}{Style.RESET_ALL}")
        else:
            get_logger().info(f"{Fore.WHITE}ℹ️ [HUGGING FACE] Integration disabled{Style.RESET_ALL}")

    def _log_initialization_status(self):
        """Log final initialization status with clean design"""
        protection_status = "ENABLED" if self.enable_crash_protection else "DISABLED"
        status_emoji = "🚀" if self.enable_crash_protection else "⚠️"
        
        get_logger().info(
            f"{Fore.GREEN}{status_emoji} [SWARM MANAGER] Initialized successfully:\n"
            f"   🤖 Model: {self.model_display_name}\n"
            f"   🐾 Agent: {self.animal_name}\n"
            f"   📍 Peer ID: {self.peer_id}...\n"
            f"   🔄 Starting Round: {self.state.round}\n"
            f"   ⏰ Submit Period: {self.submit_period} hours\n"
            f"   🎮 PRG Game: {'Enabled' if self.prg_game else 'Disabled'}\n"
            f"   🛡️ Crash Protection: {protection_status}{Style.RESET_ALL}"
        )

    def _on_dht_restart_event(self, event_type: str, reason: str):
        """Handle DHT restart events with clean logging"""
        if event_type == "restart_completed":
            get_logger().info(f"{Fore.GREEN}✅ [DHT RESTART] Completed - {reason}{Style.RESET_ALL}")
        elif event_type == "restart_failed":
            get_logger().error(f"{Fore.RED}❌ [DHT RESTART] Failed - {reason}{Style.RESET_ALL}")

    def _execute_pending_restart(self):
        """Execute pending DHT restart with clean logging"""
        if not (self.training_state_manager and self.training_state_manager._restart_requested):
            return
            
        reason = self.training_state_manager._restart_reason
        
        get_logger().info(f"{Fore.YELLOW}🔄 [DHT RESTART] Executing: {reason}{Style.RESET_ALL}")
        
        try:
            if hasattr(self.communication, 'perform_coordinated_restart'):
                self.communication.perform_coordinated_restart(reason)
                self.training_state_manager.acknowledge_restart()
                
                get_logger().info(f"{Fore.GREEN}✅ [DHT RESTART] Completed successfully{Style.RESET_ALL}")
            else:
                get_logger().error(f"{Fore.RED}❌ [DHT RESTART] Method not found{Style.RESET_ALL}")
                self.training_state_manager.acknowledge_restart()
                
        except Exception as e:
            get_logger().error(f"{Fore.RED}❌ [DHT RESTART] Execution failed: {e}{Style.RESET_ALL}")
            self.training_state_manager.acknowledge_restart()

    def _safe_blockchain_submit(self, signal_by_agent):
        """Thread-safe blockchain submit with clean logging"""
        if not self.enable_crash_protection or not self.training_state_manager:
            return self._try_submit_to_chain(signal_by_agent)
        
        # Enter critical section
        self.training_state_manager.enter_critical_section("blockchain_submit")
        
        try:
            return self._try_submit_to_chain(signal_by_agent)
        except Exception as e:
            get_logger().error(f"{Fore.RED}❌ [BLOCKCHAIN SUBMIT] Failed: {e}{Style.RESET_ALL}")
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
        """Submit results to blockchain with clean logging"""
        elapsed_hours = (time.time() - self.time_since_submit) / 3600
        
        if elapsed_hours > self.submit_period:
            try:
                get_logger().info(f"{Fore.CYAN}⛓️ [BLOCKCHAIN] Submitting round {self.state.round}...{Style.RESET_ALL}")
                
                points = int(self.batched_signals)
                get_logger().info(f"{Fore.BLUE}📊 [BLOCKCHAIN] Submitting {points} points{Style.RESET_ALL}")
                
                # Submit reward and winners
                self.coordinator.submit_reward(self.state.round, 0, points, self.peer_id)
                
                if signal_by_agent:
                    max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                    winner_name = get_name_from_peer_id(max_agent, True)
                    get_logger().info(f"{Fore.YELLOW}👑 [BLOCKCHAIN] Round winner: {winner_name} ({max_signal} points){Style.RESET_ALL}")
                else:
                    max_agent, max_signal = self.peer_id, points
                
                self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)
                
                # Reset counters
                self.batched_signals = 0.0
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                
                get_logger().info(f"{Fore.GREEN}✅ [BLOCKCHAIN] Submission successful{Style.RESET_ALL}")
                
            except Exception as e:
                get_logger().error(f"{Fore.RED}❌ [BLOCKCHAIN] Submission failed: {e}{Style.RESET_ALL}")
                get_logger().exception("Full blockchain submission error")
        else:
            remaining_minutes = (self.submit_period - elapsed_hours) * 60
            
            # Log every 30 minutes when waiting
            if not hasattr(self, '_last_waiting_log'):
                self._last_waiting_log = 0
            
            if time.time() - self._last_waiting_log > 1200:  # 30 minutes
                get_logger().info(f"{Fore.WHITE}⏳ [BLOCKCHAIN] Next submit in {remaining_minutes:.0f} minutes{Style.RESET_ALL}")
                self._last_waiting_log = time.time()

    def _hook_after_rewards_updated(self):
        """Handle reward updates with clean logging"""
        # Set training phase
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.MODEL_UPDATE)
        
        signal_by_agent = self._get_total_rewards_by_agent()
        old_signals = self.batched_signals
        self.batched_signals += self._get_my_rewards(signal_by_agent)
        
        # Log reward updates with clean format
        reward_gained = self.batched_signals - old_signals
        if reward_gained > 0:
            get_logger().info(f"{Fore.GREEN}💎 [REWARDS] Gained +{reward_gained:.1f} points (Total: {int(self.batched_signals)}){Style.RESET_ALL}")
        
        # Submit to blockchain
        self._safe_blockchain_submit(signal_by_agent)
        
        # Reset to idle phase
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.IDLE)

    def _hook_after_round_advanced(self):
        """Handle round advancement with clean logging"""
        self.round_counter += 1
        
        # Enhanced new round logging with detailed info and bright colors
        get_logger().info(
            f"{Fore.MAGENTA}{Style.BRIGHT}🚀 [ROUND ADVANCED] NEW ROUND STARTED! 🚀\n"
            f"{Fore.CYAN}   📍 Round: {self.state.round}\n"
            f"{Fore.GREEN}   🏆 Total Rounds: {self.round_counter}\n"
            f"{Fore.YELLOW}   💎 Pending Points: {int(self.batched_signals)}\n"
            f"{Fore.BLUE}   🤖 Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        # Log system health periodically
        self._log_system_health()
        
        # PRG Game logic with enhanced logging
        if self.prg_game:
            if self.training_state_manager:
                self.training_state_manager.set_phase(TrainingPhase.PRG_GAME)
            
            get_logger().info(f"{Fore.CYAN}🎮 [PRG GAME] Running game logic...{Style.RESET_ALL}")
            
            try:
                prg_history_dict = self.prg_module.prg_history_dict
                results_dict = self.trainer.play_prg_game_logits(prg_history_dict)
                self.prg_module.play_prg_game(results_dict, self.peer_id)
                
                get_logger().info(f"{Fore.GREEN}✅ [PRG GAME] Completed successfully{Style.RESET_ALL}")
            except Exception as e:
                get_logger().error(f"{Fore.RED}❌ [PRG GAME] Failed: {e}{Style.RESET_ALL}")
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
        """Handle game completion with clean logging"""
        get_logger().info(f"{Fore.MAGENTA}🎉 [GAME COMPLETION] Final cleanup...{Style.RESET_ALL}")
        
        # Log final health status
        if self.enable_crash_protection:
            self._log_comprehensive_health_status()
        
        # Final model save
        self._save_to_hf()
        
        # Clean shutdown
        if hasattr(self.communication, 'shutdown'):
            self.communication.shutdown()
            get_logger().info(f"{Fore.GREEN}✅ [SHUTDOWN] System shutdown completed{Style.RESET_ALL}")

    def _save_to_hf(self):
        """Save model to Hugging Face Hub with clean logging"""
        if (self.hf_token not in [None, "None"] and 
            self.state.round % self.hf_push_frequency == 0):
            
            get_logger().info(f"{Fore.CYAN}📤 [HUGGING FACE] Uploading model (Round {self.state.round})...{Style.RESET_ALL}")
            
            try:
                repo_id = getattr(self.trainer.args, 'hub_model_id', None) or Path(self.trainer.args.output_dir).name
                
                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=["rl-swarm", "genrl-swarm", "grpo", "gensyn", f"I am {self.animal_name}"]
                )
                
                get_logger().info(f"{Fore.GREEN}✅ [HUGGING FACE] Upload successful to {repo_id}{Style.RESET_ALL}")
                
            except Exception as e:
                get_logger().error(f"{Fore.RED}❌ [HUGGING FACE] Upload failed: {e}{Style.RESET_ALL}")

    def agent_block(self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15):
        """Enhanced agent blocking with clean logging and centralized restart execution"""
        
        # Set idle phase - this is the safest time for DHT restarts
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.IDLE)
        
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        
        get_logger().info(f"{Fore.YELLOW}⏳ [AGENT BLOCK] Waiting for swarm round advancement ({self.animal_name}){Style.RESET_ALL}")
        
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
                    get_logger().debug(f"{Fore.YELLOW}⚠️ [ROUND FETCH] Failed: {e}{Style.RESET_ALL}")
                    fetch_log_time = curr_time
                time.sleep(check_interval)
                continue

            # Check if we can advance to next round
            if round_num >= self.state.round:
                get_logger().info(f"{Fore.GREEN}🚀 [ROUND ADVANCE] Joining round {round_num}{Style.RESET_ALL}")
                check_backoff = check_interval  # Reset backoff
                self.state.round = round_num
                return
            else:
                get_logger().info(f"{Fore.WHITE}ℹ️ [ROUND STATUS] Already finished round {round_num}. Next check in {check_backoff}s{Style.RESET_ALL}")
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            # Check for final round
            if round_num == self.max_round - 1:
                get_logger().info(f"{Fore.MAGENTA}🎉 [TRAINING COMPLETE] Reached maximum round: {self.max_round}{Style.RESET_ALL}")
                return

        get_logger().info(f"{Fore.YELLOW}🕐 [TRAINING TIMEOUT] After {self.train_timeout}s{Style.RESET_ALL}")

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
        """Log system health status periodically with clean format"""
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
        
        # Create clean health status
        emergency_text = f" | Emergency: {'ACTIVE' if emergency_mode else 'OFF'}" if emergency_mode else ""
        
        get_logger().info(
            f"{Fore.CYAN}💓 [HEALTH] Phase: {training_phase} | "
            f"DHT: {dht_mode} | Restarts: {restart_count}{emergency_text}{Style.RESET_ALL}"
        )
        
        if restart_count > 30:
            get_logger().warning(f"{Fore.YELLOW}⚠️ [HEALTH] High restart count detected: {restart_count}{Style.RESET_ALL}")

    def _log_comprehensive_health_status(self):
        """Log detailed health status with clean format"""
        if not self.enable_crash_protection:
            return
            
        status = self.get_comprehensive_health_status()
        
        get_logger().info(f"{Fore.MAGENTA}🛡️ [COMPREHENSIVE HEALTH STATUS]{Style.RESET_ALL}")
        
        # Manager info
        manager_info = status.get("manager_info", {})
        get_logger().info(
            f"{Fore.WHITE}🤖 [AGENT STATUS] Agent: {manager_info.get('animal_name')} | "
            f"Round: {manager_info.get('round')} | Points: {manager_info.get('batched_signals')} | "
            f"Total Rounds: {manager_info.get('round_counter', 0)}{Style.RESET_ALL}"
        )
        
        # Training state
        training_state = status.get("training_state", {})
        if training_state:
            get_logger().info(
                f"{Fore.CYAN}🧠 [TRAINING STATE] Phase: {training_state.get('current_phase', 'unknown')} | "
                f"Total Restarts: {training_state.get('total_restarts', 0)} | "
                f"Critical Sections: {training_state.get('critical_sections', 0)}{Style.RESET_ALL}"
            )
        
        # DHT status
        dht_backend = status.get("dht_backend", {})
        if dht_backend:
            emergency_status = "ACTIVE" if dht_backend.get('emergency_mode', False) else "OFF"
            get_logger().info(
                f"{Fore.BLUE}🌐 [DHT BACKEND] Mode: {dht_backend.get('mode', 'unknown')} | "
                f"Emergency: {emergency_status} | "
                f"Connection: {dht_backend.get('connection_status', 'unknown')}{Style.RESET_ALL}"
            )


# Backward compatibility alias
SwarmGameManager = CrashSafeSwarmGameManager


# Factory function for enhanced manager creation
def create_crash_safe_swarm_manager(**kwargs):
    """Create SwarmGameManager with crash protection and clean logging"""
    
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


# Emergency control functions with clean logging
def emergency_disable_crash_protection(manager):
    """Emergency function to disable crash protection with clean logging"""
    if hasattr(manager, 'enable_crash_protection'):
        manager.enable_crash_protection = False
        
        get_logger().warning(f"{Fore.RED}❌ [EMERGENCY] Crash protection DISABLED{Style.RESET_ALL}")
        
        if hasattr(manager.communication, 'auto_restart_enabled'):
            manager.communication.auto_restart_enabled = False


def get_system_health_report(manager) -> str:
    """Get formatted system health report with clean format"""
    if not hasattr(manager, 'get_comprehensive_health_status'):
        return "Health monitoring not available"
        
    status = manager.get_comprehensive_health_status()
    
    # Create clean health report
    report_lines = []
    
    # Manager status
    manager_info = status.get("manager_info", {})
    report_lines.append(f"🤖 Agent: {manager_info.get('animal_name', 'unknown')}")
    report_lines.append(f"🏆 Round: {manager_info.get('round', 0)}")
    report_lines.append(f"💎 Points: {manager_info.get('batched_signals', 0)}")
    
    # Training state
    training_state = status.get("training_state", {})
    if training_state:
        report_lines.append(f"⚙️ Phase: {training_state.get('current_phase', 'unknown')}")
        report_lines.append(f"🔄 Restarts: {training_state.get('total_restarts', 0)}")
    
    # DHT status
    dht_backend = status.get("dht_backend", {})
    if dht_backend:
        report_lines.append(f"🌐 DHT Mode: {dht_backend.get('mode', 'unknown')}")
        
        if dht_backend.get('emergency_mode', False):
            report_lines.append("⚠️ DHT Emergency Mode: ACTIVE")
    
    return "\n".join([f"{Fore.CYAN}=== SYSTEM HEALTH REPORT ==={Style.RESET_ALL}"] + 
                     [f"{Fore.WHITE}{line}{Style.RESET_ALL}" for line in report_lines] +
                     [f"{Fore.CYAN}========================{Style.RESET_ALL}"])


# Additional utility functions for clean logging
def log_performance_metrics(manager, metrics: Dict[str, Any]):
    """Log performance metrics with clean format"""
    metric_parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key.lower() in ['accuracy', 'score', 'reward']:
                emoji = "🏆"
            elif key.lower() in ['time', 'duration', 'latency']:
                emoji = "🕐"
            elif key.lower() in ['memory', 'cpu', 'gpu']:
                emoji = "⚙️"
            else:
                emoji = "📊"
            
            metric_parts.append(f"{key}: {value} {emoji}")
        else:
            metric_parts.append(f"{key}: {value}")
    
    get_logger().info(f"{Fore.CYAN}📊 [PERFORMANCE] {' | '.join(metric_parts)}{Style.RESET_ALL}")


def log_training_progress(manager, current_step: int, total_steps: int, loss: float = None):
    """Log training progress with clean progress indication"""
    percentage = (current_step / total_steps * 100) if total_steps > 0 else 0
    
    progress_parts = [f"Step {current_step}/{total_steps} ({percentage:.1f}%)"]
    
    if loss is not None:
        progress_parts.append(f"Loss: {loss:.4f}")
    
    progress_parts.append(f"Agent: {manager.animal_name}")
    
    get_logger().info(f"{Fore.BLUE}🚀 [TRAINING PROGRESS] {' | '.join(progress_parts)}{Style.RESET_ALL}")
