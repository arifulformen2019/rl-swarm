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

# Colorful logging
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class MockFore:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = ""
    class MockStyle:
        RESET_ALL = ""
    Fore = MockFore()
    Style = MockStyle()


class CrashSafeSwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """
    Enhanced SwarmGameManager with comprehensive crash protection and automatic DHT restart.
    Features training state management, coordinated restarts, and emergency fallbacks.
    """

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
        # NEW: Crash protection parameters
        enable_crash_protection: bool = True,
        enable_dht_auto_restart: bool = True,
        memory_threshold_mb: int = 1800,
        restart_interval_minutes: int = 30,
        max_auto_restarts: int = 15,
        health_check_interval: int = 60,  # seconds
        **kwargs,
    ):

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
        # STORE CRASH PROTECTION PARAMETERS AS INSTANCE ATTRIBUTES
        self.memory_threshold_mb = memory_threshold_mb
        self.restart_interval_minutes = restart_interval_minutes
        self.max_auto_restarts = max_auto_restarts
        self.enable_crash_protection = enable_crash_protection
        self.health_check_interval = health_check_interval
        
        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # --- CRASH PROTECTION INITIALIZATION ---
        self.enable_crash_protection = enable_crash_protection
        self.health_check_interval = health_check_interval
        self._last_health_log_time = 0
        
        if self.enable_crash_protection:
            # Initialize training state manager
            self.training_state_manager = TrainingStateManager()
            
            # DEBUG: Log parameter types being passed to backend
            get_logger().debug(f"Manager crash protection params:")
            get_logger().debug(f"  memory_threshold_mb: {type(self.memory_threshold_mb)} = {self.memory_threshold_mb}")
            get_logger().debug(f"  restart_interval_minutes: {type(self.restart_interval_minutes)} = {self.restart_interval_minutes}")
            get_logger().debug(f"  max_auto_restarts: {type(self.max_auto_restarts)} = {self.max_auto_restarts}")
            
            # Register with DHT backend for coordination
            self.communication.register_training_state_manager(self.training_state_manager)
            self.communication.set_restart_callback(self._on_dht_restart_event)
            
            # POTENTIAL FIX: Ensure backend has correct types
            if hasattr(self.communication, 'memory_threshold_mb'):
                self.communication.memory_threshold_mb = float(self.memory_threshold_mb)
            if hasattr(self.communication, 'restart_interval_minutes'):
                self.communication.restart_interval_minutes = float(self.restart_interval_minutes)
            if hasattr(self.communication, 'max_auto_restarts'):
                self.communication.max_auto_restarts = int(self.max_auto_restarts)
            
            get_logger().info("🛡️ Crash protection enabled")
            get_logger().info(f"🔄 DHT auto-restart: {self.communication.auto_restart_enabled}")
            get_logger().info(f"💾 Memory threshold: {self.communication.memory_threshold_mb}MB")
            get_logger().info(f"⏱️ Restart interval: {self.communication.restart_interval_minutes}min")
        else:
            self.training_state_manager = None
            get_logger().warning("Crash protection disabled")

        # Peer and model setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        
        # --- VLLM INTEGRATION START ---
        # Safely get the model name first, then use it.
        model_name = "UnknownModel"
        
        # Check if we are in vLLM mode
        if hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm:
            # In vLLM mode, use the name we saved in the trainer
            model_name = getattr(self.trainer, "model_name", "vLLM_Model")
        else:
            # In standard training mode, safely access the config attribute
            config_obj = getattr(getattr(self.trainer, "model", None), "config", None)
            if config_obj:
                model_name = getattr(config_obj, "_name_or_path", "UnknownModel")
        
        # Clean model name for display
        self.model_display_name = self._clean_model_name(model_name)
        
        # Logging Setup with model name
        format_msg = f"[{self.model_display_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)

        get_logger().info(f"Using Model: {model_name}")

        # Register peer_id and get current round from the chain
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = (
            self.state.round
        )  # initialize communication module to contract's round

        # enable push to HF if token was provided
        self.hf_token = hf_token
        self.hf_push_frequency = hf_push_frequency
        if self.hf_token not in [None, "None"]:
            # This block should only run if we can actually push, which means we're NOT in vLLM mode.
            if not (hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm):
                try:
                    username = whoami(token=self.hf_token)["name"]
                    model_name_suffix = model_name.split("/")[-1]
                    hub_model_id = f"{username}/{model_name_suffix}-Gensyn-Swarm-{self.animal_name}"
                    
                    self.trainer.args.hub_model_id = hub_model_id
                    self.trainer.args.push_to_hub = True
                    self.trainer.args.hub_token = self.hf_token
                    
                    get_logger().info("Logging into Hugging Face Hub...")
                    login(self.hf_token)
                except Exception as e:
                    get_logger().warning(f"Could not set up Hugging Face push. Error: {e}")
            else:
                get_logger().info("Hugging Face push is disabled in vLLM mode.")
        # --- VLLM INTEGRATION END ---

        get_logger().info(
            f"🐱 Hello 🐈 [{self.animal_name}] 🦮 [{self.peer_id}]!"
        )
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        # Blockchain submission
        self.batched_signals = 0.0
        self.time_since_submit = time.time()  # seconds
        self.submit_period = 2.0  # hours
        self.submitted_this_round = False
        
        # Round counter for logging
        self.round_counter = 0

        # PRG Game initialization
        self.prg_module = PRGModule(log_dir, **kwargs)
        self.prg_game = self.prg_module.prg_game

        # Final initialization log
        status_emoji = "🛡️" if self.enable_crash_protection else "⚠️"
        protection_status = "PROTECTED" if self.enable_crash_protection else "UNPROTECTED"
        
        get_logger().info(
            f"{Fore.GREEN}{status_emoji} [SWARM MANAGER] Initialized successfully:\n"
            f"   🤖 Model: {self.model_display_name}\n"
            f"   🐾 Agent: {self.animal_name}\n"
            f"   📍 Peer ID: {self.peer_id}\n"
            f"   🔄 Starting Round: {self.state.round}\n"
            f"   ⏰ Submit Period: {self.submit_period} hours\n"
            f"   🎮 PRG Game: {'Enabled' if self.prg_game else 'Disabled'}\n"
            f"   🛡️ Crash Protection: {protection_status}{Style.RESET_ALL}"
        )

    def _on_dht_restart_event(self, event_type: str, reason: str):
        """Callback for DHT restart events"""
        if event_type == "restart_completed":
            get_logger().info(
                f"{Fore.GREEN}✅ [DHT RESTART] Restart completed successfully!\n"
                f"   🔄 Reason: {reason}\n"
                f"   🐾 Agent: {self.animal_name}\n"
                f"   📍 New Peer ID: {self.communication.get_id()}{Style.RESET_ALL}"
            )
        elif event_type == "restart_failed":
            get_logger().error(
                f"{Fore.RED}❌ [DHT RESTART] Restart failed!\n"
                f"   🚨 Reason: {reason}\n"
                f"   🐾 Agent: {self.animal_name}\n"
                f"   ⚠️ Status: Emergency mode activated{Style.RESET_ALL}"
            )

    def _safe_all_gather_object(self, obj):
        """Thread-safe all_gather with training state coordination"""
        
        if not self.enable_crash_protection or not self.training_state_manager:
            # Fallback to original behavior if crash protection disabled
            return self.communication.all_gather_object(obj)
        
        # Enter critical gradient sync phase
        self.training_state_manager.enter_critical_section("gradient_sync")
        
        try:
            # Check if restart is requested and handle it
            if (hasattr(self.training_state_manager, '_restart_requested') and
                self.training_state_manager._restart_requested):
                
                get_logger().info(
                    f"{Fore.CYAN}🔄 [COORDINATED RESTART] Processing DHT restart...{Style.RESET_ALL}"
                )
                
                # Perform coordinated restart via backend
                if hasattr(self.communication, 'perform_coordinated_restart'):
                    reason = self.training_state_manager._restart_reason
                    self.communication.perform_coordinated_restart(reason)
                    
                    # Acknowledge restart in state manager
                    self.training_state_manager.acknowledge_restart()
                    
                    get_logger().info(
                        f"{Fore.GREEN}✅ [COORDINATED RESTART] DHT restart completed{Style.RESET_ALL}"
                    )
                    
            # Perform the actual all_gather
            result = self.communication.all_gather_object(obj)
            return result
            
        except Exception as e:
            get_logger().error(f"{Fore.RED}❌ [SAFE GATHER] All-gather failed: {e}{Style.RESET_ALL}")
            
            # Check if this is a critical error that needs restart
            error_msg = str(e).lower()
            critical_patterns = [
                "ran out of input", "pipe", "broken", "connection",
                "timeout", "eof", "resource temporarily unavailable"
            ]
            
            if any(pattern in error_msg for pattern in critical_patterns):
                get_logger().error(
                    f"{Fore.RED}🚨 [CRITICAL ERROR] DHT error detected - requesting restart{Style.RESET_ALL}"
                )
                if self.training_state_manager:
                    self.training_state_manager.request_restart(f"All-gather error: {e}")
            
            # Return single-node fallback
            agent_id = self.communication.get_id()
            get_logger().info(
                f"{Fore.YELLOW}⚠️ [FALLBACK] Using single-node mode (agent: {agent_id}){Style.RESET_ALL}"
            )
            return {agent_id: obj}
            
        finally:
            # Always exit critical section
            if self.training_state_manager:
                self.training_state_manager.exit_critical_section("gradient_sync")

    def _safe_blockchain_submit(self, signal_by_agent):
        """Thread-safe blockchain submit"""
        
        if not self.enable_crash_protection or not self.training_state_manager:
            # Fallback to original behavior
            return self._try_submit_to_chain(signal_by_agent)
        
        # Enter critical blockchain submit phase
        self.training_state_manager.enter_critical_section("blockchain_submit")
        
        try:
            # Use original submit method
            return self._try_submit_to_chain(signal_by_agent)
            
        except Exception as e:
            get_logger().error(
                f"{Fore.RED}❌ [BLOCKCHAIN SUBMIT] Submit failed: {e}{Style.RESET_ALL}"
            )
            raise
            
        finally:
            # Always exit critical section
            if self.training_state_manager:
                self.training_state_manager.exit_critical_section("blockchain_submit")

    def _clean_model_name(self, model_name):
        """Clean model name for display"""
        if "/" in model_name:
            clean_name = model_name.split("/")[-1]
        else:
            clean_name = model_name
            
        # Remove common suffixes
        clean_suffixes = ["-Instruct", "-Chat", "-Base", "-v1", "-v2", "-v3"]
        for suffix in clean_suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        return clean_name

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot

        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        if len(signal_by_agent) == 0:
            return 0
        if self.peer_id in signal_by_agent:
            my_signal = signal_by_agent[self.peer_id]
        else:
            my_signal = 0
        my_signal = (my_signal + 1) * (my_signal > 0) + my_signal * (
            my_signal <= 0
        )
        return my_signal

    def _try_submit_to_chain(self, signal_by_agent):
        elapsed_time_hours = (time.time() - self.time_since_submit) / 3600
        
        if elapsed_time_hours > self.submit_period:
            try:
                get_logger().info(
                    f"{Fore.CYAN}🚀 [SUBMIT STARTING] Round: {self.state.round} | "
                    f"Points: {int(self.batched_signals)} | Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                # Submit reward
                self.coordinator.submit_reward(
                    self.state.round, 0, int(self.batched_signals), self.peer_id
                )
                
                # Determine winner
                if len(signal_by_agent) > 0:
                    max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                    try:
                        winner_name = get_name_from_peer_id(max_agent, True) if max_agent != self.peer_id else self.animal_name
                    except:
                        winner_name = "unknown-agent"
                else:
                    max_agent = self.peer_id
                    winner_name = self.animal_name
                    max_signal = int(self.batched_signals)

                # Submit winners
                self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [SUBMIT SUCCESS] 🎉 POINTS SUBMITTED! 🎉\n"
                    f"   💰 Points Sent: {int(self.batched_signals)}\n"
                    f"   🏆 Round Winner: {winner_name} ({max_signal} pts)\n"
                    f"   🕐 Next Submit: {self.submit_period} hours\n"
                    f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                # Reset counters
                submitted_points = int(self.batched_signals)
                self.batched_signals = 0.0
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                
                get_logger().info(
                    f"{Fore.BLUE}📊 [STATS] Total Submitted: {submitted_points} | "
                    f"Round: {self.state.round}{Style.RESET_ALL}"
                )
                
            except Exception as e:
                get_logger().error(
                    f"{Fore.RED}❌ [SUBMIT FAILED] 💥 SUBMISSION ERROR! 💥\n"
                    f"   🚨 Error: {str(e)}\n"
                    f"   💰 Points Lost: {int(self.batched_signals)}\n"
                    f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                
                get_logger().exception(
                    "Failed to submit to chain.\n"
                    "This is most likely transient and will recover.\n"
                    "There is no need to kill the program.\n"
                    "If you encounter this error, please report it to Gensyn by\n"
                    "filing a github issue here: https://github.com/gensyn-ai/rl-swarm/issues/ \n"
                    "including the full stacktrace."
                )
        else:
            remaining_hours = self.submit_period - elapsed_time_hours
            remaining_minutes = remaining_hours * 60
            
            # Only log every 30 minutes when waiting
            if not hasattr(self, '_last_waiting_log'):
                self._last_waiting_log = 0
            
            if time.time() - self._last_waiting_log > 1800:  # 30 minutes
                get_logger().info(
                    f"{Fore.YELLOW}⏳ [WAITING] Next submit in: {remaining_minutes:.0f} minutes | "
                    f"Current points: {int(self.batched_signals)} | Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                self._last_waiting_log = time.time()

    def _hook_after_rewards_updated(self):
        """Enhanced rewards update with crash protection"""
        
        # Set training phase
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.MODEL_UPDATE)
        
        signal_by_agent = self._get_total_rewards_by_agent()
        old_signals = self.batched_signals
        self.batched_signals += self._get_my_rewards(signal_by_agent)
        
        # Log reward updates
        reward_gained = self.batched_signals - old_signals
        if reward_gained > 0:
            get_logger().info(
                f"{Fore.GREEN}💰 [REWARD GAINED] +{reward_gained:.1f} points | "
                f"Total: {int(self.batched_signals)} | Agent: {self.animal_name}{Style.RESET_ALL}"
            )
        
        # Use safe blockchain submit
        self._safe_blockchain_submit(signal_by_agent)
        
        # Reset to idle
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.IDLE)

    def _hook_after_round_advanced(self):
        """Enhanced round advancement with crash protection"""
        
        self.round_counter += 1
        
        get_logger().info(
            f"{Fore.MAGENTA}🔄 [ROUND ADVANCED] 🚀 NEW ROUND STARTED! 🚀\n"
            f"   📈 Round: {self.state.round}\n"  
            f"   🎯 Total Rounds: {self.round_counter}\n"
            f"   💰 Pending Points: {int(self.batched_signals)}\n"
            f"   🐾 Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        # Log system health periodically
        self._log_system_health()
        
        # PRG Game logic with crash protection
        if self.prg_game:
            # Set PRG game phase
            if self.training_state_manager:
                self.training_state_manager.set_phase(TrainingPhase.PRG_GAME)
            
            get_logger().info(
                f"{Fore.BLUE}🎮 [PRG GAME] Starting PRG game logic | "
                f"Round: {self.state.round} | Agent: {self.animal_name}{Style.RESET_ALL}"
            )
            try:
                prg_history_dict = self.prg_module.prg_history_dict
                results_dict = self.trainer.play_prg_game_logits(prg_history_dict)
                self.prg_module.play_prg_game(results_dict, self.peer_id)
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [PRG GAME] PRG game completed successfully | "
                    f"Agent: {self.animal_name}{Style.RESET_ALL}"
                )
            except Exception as e:
                get_logger().error(
                    f"{Fore.RED}❌ [PRG GAME] PRG game failed: {str(e)} | "
                    f"Agent: {self.animal_name}{Style.RESET_ALL}"
                )
                get_logger().exception("PRG Game error details:")
            finally:
                # Reset phase
                if self.training_state_manager:
                    self.training_state_manager.set_phase(TrainingPhase.IDLE)
        
        self._save_to_hf()

        # Safe blockchain submit for final round check
        if not self.submitted_this_round:
            signal_by_agent = self._get_total_rewards_by_agent()
            self._safe_blockchain_submit(signal_by_agent)
        
        # Reset flag for next round
        self.submitted_this_round = False

        # Block until swarm round advances (this is safe for restarts)
        self.agent_block()

    def _hook_after_game(self):
        """Enhanced game end with crash protection cleanup"""
        
        get_logger().info(
            f"{Fore.GREEN}🎮 [GAME ENDED] Final save and cleanup | Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        # Final health status log
        self._log_comprehensive_health_status()
        
        # Save to HF
        self._save_to_hf()
        
        # Clean shutdown of crash protection
        if self.enable_crash_protection and hasattr(self.communication, 'shutdown'):
            get_logger().info("Shutting down crash protection systems...")
            self.communication.shutdown()

    def _save_to_hf(self):
        # This check also implicitly prevents pushes in vLLM mode because hf_token setup is skipped
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            get_logger().info(
                f"{Fore.BLUE}📤 [HF PUSH] Pushing model to Hugging Face Hub | Round: {self.state.round}{Style.RESET_ALL}"
            )
            try:
                repo_id = self.trainer.args.hub_model_id
                if repo_id is None:
                    repo_id = Path(self.trainer.args.output_dir).name

                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [HF SUCCESS] Model pushed successfully to {repo_id}{Style.RESET_ALL}"
                )
                
            except Exception as e:
                get_logger().error(f"{Fore.RED}❌ [HF FAILED] Failed to push model: {str(e)}{Style.RESET_ALL}")
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading",
                    stack_info=True,
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        """Enhanced agent block with crash protection"""
        
        # Set idle phase - safe for DHT restarts
        if self.training_state_manager:
            self.training_state_manager.set_phase(TrainingPhase.IDLE)
        
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        
        get_logger().info(
            f"{Fore.YELLOW}⏸️ [BLOCKING] Waiting for swarm round advancement... | "
            f"Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            
            # DHT health check with crash protection
            try:
                if self.communication.dht:
                    _ = self.communication.dht.get_visible_maddrs(latest=True)
            except Exception as e:
                get_logger().warning(
                    f"{Fore.YELLOW}⚠️ [DHT WARNING] Health check failed during blocking: {e}{Style.RESET_ALL}"
                )
                # Don't crash on DHT errors during blocking - this is a safe phase

            # Retrieve current round and stage with error handling
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"{Fore.YELLOW}🔍 Could not fetch round and stage: {e}. "
                        f"Next check in {check_interval}s.{Style.RESET_ALL}"
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(
                    f"{Fore.GREEN}🐝 [JOINING] Joining round: {round_num} | "
                    f"Model: {self.model_display_name}{Style.RESET_ALL}"
                )
                check_backoff = check_interval  # Reset backoff after successful round
                self.state.round = round_num  # advance to swarm's round.
                return
            else:
                get_logger().info(
                    f"{Fore.YELLOW}⏭️ Already finished round: {round_num}. "
                    f"Next check in {check_backoff}s.{Style.RESET_ALL}"
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                get_logger().info(
                    f"{Fore.MAGENTA}🏁 [FINAL ROUND] Reached maximum round: {self.max_round}{Style.RESET_ALL}"
                )
                return

        get_logger().info(
            f"{Fore.RED}⏰ [TIMEOUT] Training timed out after {self.train_timeout}s!{Style.RESET_ALL}"
        )

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
        
        # Get training state status
        if self.training_state_manager:
            status["training_state"] = self.training_state_manager.get_stats()
        
        # Get DHT backend status
        if hasattr(self.communication, 'get_auto_restart_status'):
            status["dht_auto_restart"] = self.communication.get_auto_restart_status()
            
        if hasattr(self.communication, 'get_status'):
            status["dht_backend"] = self.communication.get_status()
            
        return status

    def _log_system_health(self):
        """Log system health status periodically"""
        current_time = time.time()
        
        # Only log every health_check_interval seconds
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
        
        # Log basic health status
        get_logger().info(
            f"{Fore.CYAN}💊 [HEALTH] Phase: {training_phase} | "
            f"Mode: {dht_mode} | "
            f"Restarts: {restart_count} | "
            f"Agent: {self.animal_name}{Style.RESET_ALL}"
        )
        
        # Log warnings if needed
        if emergency_mode:
            get_logger().warning(
                f"{Fore.RED}🚨 [EMERGENCY] Emergency mode active - single-node fallback | "
                f"Agent: {self.animal_name}{Style.RESET_ALL}"
            )
            
        if restart_count > 5:
            get_logger().warning(
                f"{Fore.YELLOW}⚠️ [HIGH RESTARTS] Restart count: {restart_count} | "
                f"Agent: {self.animal_name}{Style.RESET_ALL}"
            )

    def _log_comprehensive_health_status(self):
        """Log detailed health status"""
        if not self.enable_crash_protection:
            return
            
        status = self.get_comprehensive_health_status()
        
        get_logger().info("=" * 60)
        get_logger().info("COMPREHENSIVE HEALTH STATUS")
        get_logger().info("=" * 60)
        
        # Manager info
        manager_info = status.get("manager_info", {})
        get_logger().info(f"Agent: {manager_info.get('animal_name', 'unknown')}")
        get_logger().info(f"Peer ID: {manager_info.get('peer_id', 'unknown')}")
        get_logger().info(f"Round: {manager_info.get('round', 0)}")
        get_logger().info(f"Points: {manager_info.get('batched_signals', 0)}")
        get_logger().info(f"Total Rounds: {manager_info.get('round_counter', 0)}")
        
        # Training state
        training_state = status.get("training_state", {})
        if training_state:
            get_logger().info(f"Training Phase: {training_state.get('current_phase', 'unknown')}")
            get_logger().info(f"Total Restarts: {training_state.get('total_restarts', 0)}")
            get_logger().info(f"Emergency Activations: {training_state.get('emergency_activations', 0)}")
        
        # DHT status
        dht_backend = status.get("dht_backend", {})
        if dht_backend:
            get_logger().info(f"DHT Mode: {dht_backend.get('mode', 'unknown')}")
            get_logger().info(f"DHT Active: {dht_backend.get('dht_active', False)}")
            get_logger().info(f"Emergency Mode: {dht_backend.get('emergency_mode', False)}")
        
        # Auto-restart status
        auto_restart = status.get("dht_auto_restart", {})
        if auto_restart:
            get_logger().info(f"Auto-restart Enabled: {auto_restart.get('enabled', False)}")
            get_logger().info(f"Restart Count: {auto_restart.get('restart_count', 0)}")
            get_logger().info(f"Memory Threshold: {auto_restart.get('memory_threshold_mb', 0)}MB")
        
        get_logger().info("=" * 60)

    def patch_trainer_communication(self):
        """Patch trainer's communication methods to use safe versions"""
        if not self.enable_crash_protection:
            return
            
        if hasattr(self.trainer, 'all_gather_object'):
            get_logger().info("Patching trainer communication with crash protection")
            
            # Store original method
            self.trainer._original_all_gather_object = self.trainer.all_gather_object
            
            # Replace with safe version
            self.trainer.all_gather_object = self._safe_all_gather_object
            
            get_logger().info("Trainer communication patching completed")


# For backward compatibility - alias to original name
SwarmGameManager = CrashSafeSwarmGameManager


# Factory function for creating crash-safe manager
def create_crash_safe_swarm_manager(
    coordinator,
    max_stage,
    max_round,
    game_state,
    reward_manager,
    trainer,
    data_manager,
    communication,
    role_manager=None,
    run_mode="train",
    log_dir="logs",
    hf_token=None,
    hf_push_frequency=20,
    # Enhanced DHT parameters
    enable_dht_auto_restart=True,
    memory_threshold_mb=1800,
    restart_interval_minutes=30,
    max_auto_restarts=15,
    **kwargs
):
    """
    Factory function to create SwarmGameManager with comprehensive crash protection
    
    Args:
        ... (all original SwarmGameManager args)
        enable_dht_auto_restart: Enable automatic DHT restart on memory/health issues
        memory_threshold_mb: Memory threshold for triggering restart (MB)
        restart_interval_minutes: Periodic restart interval (minutes)
        max_auto_restarts: Maximum number of automatic restarts allowed
    """
    
    # Enhanced communication kwargs for crash protection
    if hasattr(communication, 'auto_restart_enabled'):
        communication.auto_restart_enabled = enable_dht_auto_restart
        communication.memory_threshold_mb = memory_threshold_mb
        communication.restart_interval_minutes = restart_interval_minutes
        communication.max_auto_restarts = max_auto_restarts
    
    # Create manager with crash protection
    manager = CrashSafeSwarmGameManager(
        coordinator=coordinator,
        max_stage=max_stage,
        max_round=max_round,
        game_state=game_state,
        reward_manager=reward_manager,
        trainer=trainer,
        data_manager=data_manager,
        communication=communication,
        role_manager=role_manager,
        run_mode=run_mode,
        log_dir=log_dir,
        hf_token=hf_token,
        hf_push_frequency=hf_push_frequency,
        enable_crash_protection=True,
        enable_dht_auto_restart=enable_dht_auto_restart,
        memory_threshold_mb=memory_threshold_mb,
        restart_interval_minutes=restart_interval_minutes,
        max_auto_restarts=max_auto_restarts,
        **kwargs
    )
    
    # Patch trainer communication
    manager.patch_trainer_communication()
    
    return manager


# Emergency control functions
def emergency_disable_crash_protection(manager):
    """Emergency function to disable crash protection"""
    if hasattr(manager, 'enable_crash_protection'):
        manager.enable_crash_protection = False
        get_logger().warning("CRASH PROTECTION EMERGENCY DISABLED")
        
        # Disable auto-restart in backend
        if hasattr(manager.communication, 'auto_restart_enabled'):
            manager.communication.auto_restart_enabled = False
            

def get_system_health_report(manager) -> str:
    """Get formatted system health report"""
    if not hasattr(manager, 'get_comprehensive_health_status'):
        return "Health monitoring not available"
        
    status = manager.get_comprehensive_health_status()
    
    report = []
    report.append("=== SYSTEM HEALTH REPORT ===")
    
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
        
        if training_state.get('emergency_activations', 0) > 0:
            report.append(f"⚠️ Emergency activations: {training_state.get('emergency_activations', 0)}")
    
    # DHT status
    dht_backend = status.get("dht_backend", {})
    if dht_backend:
        mode = dht_backend.get('mode', 'unknown')
        report.append(f"DHT Mode: {mode}")
        
        if dht_backend.get('emergency_mode', False):
            report.append("🚨 DHT Emergency Mode: ACTIVE")
        
        if not dht_backend.get('dht_active', True):
            report.append("⚠️ DHT Status: INACTIVE")
    
    # Auto-restart status
    auto_restart = status.get("dht_auto_restart", {})
    if auto_restart:
        restart_count = auto_restart.get('restart_count', 0)
        report.append(f"Auto-restarts: {restart_count}/{auto_restart.get('max_restarts', 0)}")
        
        if restart_count > 5:
            report.append("⚠️ High restart count detected")
    
    report.append("========================")
    
    return "\n".join(report)


# Usage example for main training script
def main_training_example():
    """
    Example of how to use the crash-safe SwarmGameManager in main training script
    """
    
    # Import required modules (adjust imports based on your setup)
    from genrl.communication.hivemind.hivemind_backend import create_enhanced_hivemind_backend
    
    # Create enhanced DHT backend with auto-restart
    communication = create_enhanced_hivemind_backend(
        initial_peers=["..."],  # Your initial peers
        auto_restart_enabled=True,
        memory_threshold_mb=1800,  # 1.8GB threshold
        restart_interval_minutes=30,  # Restart every 30 minutes
        max_auto_restarts=15,  # Allow up to 15 restarts
        enable_robust_mode=True,
        health_check_interval=30,
    )
    
    # Create crash-safe SwarmGameManager
    manager = create_crash_safe_swarm_manager(
        coordinator=coordinator,  # Your SwarmCoordinator
        max_stage=max_stage,
        max_round=max_round,
        game_state=game_state,
        reward_manager=reward_manager,
        trainer=trainer,
        data_manager=data_manager,
        communication=communication,  # Use the enhanced backend
        role_manager=role_manager,
        run_mode="train",
        log_dir="logs",
        hf_token=hf_token,
        hf_push_frequency=20,
        # Crash protection settings
        enable_dht_auto_restart=True,
        memory_threshold_mb=1800,
        restart_interval_minutes=30,
        max_auto_restarts=15,
    )
    
    # Start training with crash protection
    try:
        manager.run()  # or whatever your main training loop is
        
    except KeyboardInterrupt:
        get_logger().info("Training interrupted by user")
        
    except Exception as e:
        get_logger().error(f"Training failed: {e}")
        
        # Get health report before exit
        health_report = get_system_health_report(manager)
        get_logger().info(f"Final health status:\n{health_report}")
        
        raise
        
    finally:
        # Clean shutdown
        if hasattr(manager, 'communication') and hasattr(manager.communication, 'shutdown'):
            manager.communication.shutdown()
            
        get_logger().info("Training completed - crash protection disabled")


# Integration guide for existing codebases
def integration_guide():
    """
    Guide for integrating crash protection into existing SwarmGameManager code
    
    STEP 1: Update imports
    Replace:
        from your_module import SwarmGameManager
    With:
        from your_module import CrashSafeSwarmGameManager as SwarmGameManager
        # OR use the factory function:
        from your_module import create_crash_safe_swarm_manager
    
    STEP 2: Update DHT backend creation
    Replace:
        communication = HivemindBackend(...)
    With:
        communication = create_enhanced_hivemind_backend(
            ...,  # your existing parameters
            auto_restart_enabled=True,
            memory_threshold_mb=1800,
            restart_interval_minutes=30,
            max_auto_restarts=15,
        )
    
    STEP 3: Update manager creation (if using factory)
    Replace:
        manager = SwarmGameManager(...)
    With:
        manager = create_crash_safe_swarm_manager(...)
    
    STEP 4: Add error handling in main loop
    try:
        manager.run()
    except Exception as e:
        health_report = get_system_health_report(manager)
        logger.error(f"Training failed: {e}")
        logger.info(f"Health report:\n{health_report}")
        raise
    finally:
        if hasattr(manager.communication, 'shutdown'):
            manager.communication.shutdown()
    
    STEP 5: Optional - Add health monitoring
    # In your training loop, periodically check:
    if hasattr(manager, 'get_comprehensive_health_status'):
        status = manager.get_comprehensive_health_status()
        if status.get('dht_backend', {}).get('emergency_mode', False):
            logger.warning("DHT in emergency mode - training degraded")
    """
    pass


# Configuration presets for different use cases
class CrashProtectionPresets:
    """Predefined configuration presets for different scenarios"""
    
    @staticmethod
    def conservative():
        """Conservative settings - minimal restarts, high stability"""
        return {
            'auto_restart_enabled': True,
            'memory_threshold_mb': 2500,  # Higher threshold
            'restart_interval_minutes': 60,  # Restart every hour
            'max_auto_restarts': 8,  # Fewer restarts allowed
            'health_check_interval': 60,  # Less frequent checks
        }
    
    @staticmethod
    def aggressive():
        """Aggressive settings - frequent restarts, maximum uptime"""
        return {
            'auto_restart_enabled': True,
            'memory_threshold_mb': 1500,  # Lower threshold
            'restart_interval_minutes': 20,  # Restart every 20 minutes
            'max_auto_restarts': 25,  # More restarts allowed
            'health_check_interval': 15,  # Frequent checks
        }
    
    @staticmethod
    def balanced():
        """Balanced settings - good for most use cases"""
        return {
            'auto_restart_enabled': True,
            'memory_threshold_mb': 1800,  # Moderate threshold
            'restart_interval_minutes': 30,  # Restart every 30 minutes
            'max_auto_restarts': 15,  # Reasonable restart limit
            'health_check_interval': 30,  # Regular checks
        }
    
    @staticmethod
    def debug():
        """Debug settings - minimal interference for debugging"""
        return {
            'auto_restart_enabled': False,  # Disabled for debugging
            'memory_threshold_mb': 3000,  # High threshold
            'restart_interval_minutes': 120,  # Rare restarts
            'max_auto_restarts': 3,  # Very few restarts
            'health_check_interval': 120,  # Infrequent checks
        }


def create_manager_with_preset(preset_name: str, **kwargs):
    """Create manager with predefined crash protection preset"""
    
    presets = {
        'conservative': CrashProtectionPresets.conservative(),
        'aggressive': CrashProtectionPresets.aggressive(), 
        'balanced': CrashProtectionPresets.balanced(),
        'debug': CrashProtectionPresets.debug(),
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    preset_config = presets[preset_name]
    
    # Merge preset with user kwargs (user kwargs take precedence)
    config = {**preset_config, **kwargs}
    
    get_logger().info(f"Creating SwarmGameManager with '{preset_name}' crash protection preset")
    
    return create_crash_safe_swarm_manager(**config)


# Monitoring and alerting utilities
class HealthMonitor:
    """Utility class for monitoring SwarmGameManager health"""
    
    def __init__(self, manager, alert_threshold_restarts=10):
        self.manager = manager
        self.alert_threshold_restarts = alert_threshold_restarts
        self.last_check_time = time.time()
        self.alerts_sent = set()
        
    def check_health(self) -> Dict[str, Any]:
        """Check current health and return status"""
        if not hasattr(self.manager, 'get_comprehensive_health_status'):
            return {'status': 'monitoring_unavailable'}
            
        status = self.manager.get_comprehensive_health_status()
        current_time = time.time()
        
        # Check for alerts
        restart_count = status.get('dht_auto_restart', {}).get('restart_count', 0)
        emergency_mode = status.get('dht_backend', {}).get('emergency_mode', False)
        
        alerts = []
        
        # High restart count alert
        if restart_count >= self.alert_threshold_restarts:
            alert_key = f"high_restarts_{restart_count}"
            if alert_key not in self.alerts_sent:
                alerts.append({
                    'type': 'high_restart_count',
                    'message': f"High restart count: {restart_count}",
                    'severity': 'warning'
                })
                self.alerts_sent.add(alert_key)
        
        # Emergency mode alert
        if emergency_mode:
            alert_key = "emergency_mode"
            if alert_key not in self.alerts_sent:
                alerts.append({
                    'type': 'emergency_mode',
                    'message': "DHT in emergency mode - single-node fallback active",
                    'severity': 'critical'
                })
                self.alerts_sent.add(alert_key)
        
        # Reset alerts if conditions cleared
        if restart_count < self.alert_threshold_restarts:
            self.alerts_sent.discard(f"high_restarts_{restart_count}")
            
        if not emergency_mode:
            self.alerts_sent.discard("emergency_mode")
        
        self.last_check_time = current_time
        
        return {
            'status': status,
            'alerts': alerts,
            'check_time': current_time
        }
    
    def get_health_summary(self) -> str:
        """Get a brief health summary string"""
        health = self.check_health()
        
        if 'status' not in health or health['status'] == 'monitoring_unavailable':
            return "Health monitoring unavailable"
            
        status = health['status']
        
        # Extract key metrics
        manager_info = status.get('manager_info', {})
        dht_backend = status.get('dht_backend', {})
        auto_restart = status.get('dht_auto_restart', {})
        
        agent_name = manager_info.get('animal_name', 'unknown')
        round_num = manager_info.get('round', 0)
        dht_mode = dht_backend.get('mode', 'unknown')
        restart_count = auto_restart.get('restart_count', 0)
        
        summary = f"Agent: {agent_name} | Round: {round_num} | Mode: {dht_mode} | Restarts: {restart_count}"
        
        # Add alert indicators
        if health.get('alerts'):
            alert_types = [alert['type'] for alert in health['alerts']]
            summary += f" | ALERTS: {', '.join(alert_types)}"
        
        return summary
