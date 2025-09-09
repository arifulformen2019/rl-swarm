from typing import Any, Optional, List
import gc
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from genrl.data import DataManager
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule
from reasoning_gym.utils import SYSTEM_PROMPTS
from rgym_exp.src.utils.judge_client import JudgeClient
from rgym_exp.src.prg_module import PRGGameStatus

# INLINE ROBUST COMMUNICATION - No external dependencies
class EmergencyTrainingWrapper:
    """Emergency wrapper to prevent training crashes from communication errors"""
    
    def __init__(self, communication_backend):
        self.backend = communication_backend
        self.emergency_mode = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.total_emergency_calls = 0
        
    def safe_all_gather(self, obj):
        """Ultra-safe wrapper around all_gather_object"""
        try:
            if self.emergency_mode:
                self.total_emergency_calls += 1
                if self.total_emergency_calls % 100 == 0:
                    get_logger().warning(f"Emergency mode: {self.total_emergency_calls} single-node calls")
                return {self.backend.get_id(): obj}
                
            result = self.backend.all_gather_object(obj)
            
            # Reset error count on success
            if self.consecutive_errors > 0:
                get_logger().info(f"Communication recovered after {self.consecutive_errors} errors")
                self.consecutive_errors = 0
                
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.consecutive_errors += 1
            
            get_logger().error(f"EMERGENCY CATCH #{self.consecutive_errors}: {error_msg}")
            
            # Check for critical errors
            critical_patterns = [
                "ran out of input", "pipe", "broken", "connection", "timeout", 
                "eof", "resource temporarily unavailable", "blocking"
            ]
            
            if any(pattern in error_msg.lower() for pattern in critical_patterns):
                get_logger().error("Critical communication error detected - enabling emergency mode")
                self.emergency_mode = True
                
            # Too many consecutive errors
            if self.consecutive_errors >= self.max_consecutive_errors:
                get_logger().error(f"Too many consecutive errors ({self.consecutive_errors}) - emergency mode")
                self.emergency_mode = True
                
            # Always return single-node result to continue training
            return {self.backend.get_id(): obj}

    def all_gather_object(self, obj):
        """Direct wrapper around safe_all_gather"""
        return self.safe_all_gather(obj)

    def get_id(self):
        """Get ID from backend"""
        return self.backend.get_id()

    def shutdown(self):
        """Shutdown backend"""
        if hasattr(self.backend, 'shutdown'):
            self.backend.shutdown()


class FallbackBackend:
    """Simple fallback backend for single-node operation"""
    
    def __init__(self):
        self.agent_id = f"fallback_{os.getpid()}"
        self.mode = "single_node_fallback"
        
    def all_gather_object(self, obj):
        return {self.agent_id: obj}
        
    def safe_all_gather(self, obj):
        return {self.agent_id: obj}
        
    def get_id(self):
        return self.agent_id
        
    def get_training_mode(self):
        return self.mode
        
    def shutdown(self):
        pass


def create_robust_communication_wrapper(existing_backend):
    """Create robust wrapper around existing communication backend"""
    
    if existing_backend is None:
        get_logger().warning("No existing backend - creating fallback")
        return FallbackBackend()
    
    # Check if it already has robust features
    if hasattr(existing_backend, 'safe_all_gather'):
        get_logger().info("Backend already has robust features")
        return existing_backend
    
    # Wrap with emergency handler
    get_logger().info("Wrapping existing backend with emergency handler")
    return EmergencyTrainingWrapper(existing_backend)


def emergency_disable_dht():
    """Emergency function to disable DHT completely"""
    os.environ['DISABLE_DHT'] = 'true'
    os.environ['FORCE_SINGLE_NODE'] = 'true'
    get_logger().warning("DHT EMERGENCY DISABLED - all communication will use single-node mode")


# Enhanced trainer prompts
PRG_SYSTEM_PROMPT = """Given a question, hints, and possible answers, your task is to answer the question by thinking step-by-step in a clear and specific manner for 1 line only.
Your answer MUST be one of the possible answers. Provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer.
"""

PRG_SYSTEM_PROMPT_NO_THINKING = """Given a question, hints, and possible answers, your task is to answer the question.
Your answer MUST be one of the possible answers. Give your answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning at all, provide only the final answer in the answer tag.
"""


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Enhanced GRPO Trainer with built-in robust communication handling.
    Works with or without external robust communication library.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module with robust communication.
        """
        # Initialize robust communication first
        self._init_robust_communication(kwargs)
        
        # Check and fix quantization for existing models
        if models:
            for i, model in enumerate(models):
                is_quantized = self._is_model_quantized(model)
                
                if not is_quantized:
                    get_logger().warning(f"Model {i} is not quantized, reloading...")
                    try:
                        models[i] = self._reload_with_quantization(model, kwargs)
                    except Exception as e:
                        get_logger().error(f"Failed to reload model {i}: {e}")

        # Fallback: load model with quantization if no models provided
        if not models:
            model_id = kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            models = [model]
            self.tokenizer = tokenizer
        
        super().__init__(models, **kwargs)
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        
        # Communication error tracking
        self._communication_errors = 0
        self._emergency_mode_enabled = False
        self._last_successful_gather = 0
        self._step_counter = 0

    def _init_robust_communication(self, kwargs):
        """Initialize robust communication backend"""
        
        # Always initialize a robust backend - even in single-node
        existing_backend = None
        
        # Try to get communication backend from parent or kwargs
        if hasattr(self, 'communication'):
            existing_backend = self.communication
        elif 'communication' in kwargs:
            existing_backend = kwargs['communication']
        
        # Check world size to determine if we expect distributed training
        world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        
        if existing_backend is not None:
            get_logger().info(f"Found existing communication backend, wrapping with robust features")
            self._robust_backend = create_robust_communication_wrapper(existing_backend)
        elif world_size > 1:
            get_logger().info(f"Distributed environment detected (world_size={world_size}) but no backend found - using fallback")
            self._robust_backend = FallbackBackend()
        else:
            get_logger().info("Single-node environment - using fallback backend")
            self._robust_backend = FallbackBackend()

    def robust_all_gather(self, data, step_info=None):
        """
        Public method for robust distributed gathering.
        This is the main method to call from training loops.
        """
        
        if not self._robust_backend:
            return {"single_node": data}
        
        # Handle step info
        if isinstance(step_info, dict):
            step_num = step_info.get('step', self._step_counter)
        elif isinstance(step_info, int):
            step_num = step_info
        else:
            step_num = self._step_counter
            
        self._step_counter = max(self._step_counter, step_num) + 1
        
        try:
            # Try safe_all_gather first (emergency wrapper)
            if hasattr(self._robust_backend, 'safe_all_gather'):
                result = self._robust_backend.safe_all_gather(data)
                
                # Log communication status periodically
                if step_num % 1000 == 0 and step_num > 0:
                    self._log_communication_status(step_num, result)
                
                # Reset error count on successful distributed gather
                if len(result) > 1 and self._communication_errors > 0:
                    get_logger().info(f"Communication recovered at step {step_num} - back to distributed mode")
                    self._communication_errors = 0
                    self._last_successful_gather = step_num
                elif len(result) > 1:
                    self._last_successful_gather = step_num
                
                return result
                
            # Fallback to direct backend
            elif hasattr(self._robust_backend, 'all_gather_object'):
                return self._robust_backend.all_gather_object(data)
                
            else:
                get_logger().debug(f"No communication method available at step {step_num}")
                return {self._robust_backend.get_id(): data}
                
        except Exception as e:
            return self._handle_communication_error(e, data, step_num)

    def _handle_communication_error(self, error, data, step_num):
        """Handle communication errors gracefully"""
        
        self._communication_errors += 1
        error_msg = str(error)
        
        # Log error but don't spam logs
        if self._communication_errors <= 5 or self._communication_errors % 100 == 0:
            get_logger().error(f"Step {step_num}: Communication failed ({self._communication_errors}): {error_msg}")
        
        # Check for critical communication errors
        critical_patterns = [
            "ran out of input", "pipe", "connection", "dht", "hivemind", 
            "eof", "resource temporarily unavailable", "blocking"
        ]
        
        if any(pattern in error_msg.lower() for pattern in critical_patterns):
            if self._communication_errors <= 3:  # Only log first few times
                get_logger().warning(f"Critical communication error detected at step {step_num}")
            self._enable_emergency_mode()
        
        # Ultimate fallback - single node processing
        agent_id = getattr(self._robust_backend, 'get_id', lambda: f"emergency_{os.getpid()}")()
        return {agent_id: data}

    def _enable_emergency_mode(self):
        """Enable emergency mode to prevent further communication failures"""
        
        if not self._emergency_mode_enabled:
            self._emergency_mode_enabled = True
            get_logger().warning("EMERGENCY MODE ENABLED - switching to single-node training")
            
            # Try to enable emergency mode in backend
            if hasattr(self._robust_backend, 'backend'):
                if hasattr(self._robust_backend.backend, '_emergency_mode'):
                    self._robust_backend.backend._emergency_mode = True
            
            # Set environment flag for immediate effect
            emergency_disable_dht()

    def _log_communication_status(self, step_num, gathered_results):
        """Log communication status"""
        
        # Get backend status if available
        mode = getattr(self._robust_backend, 'get_training_mode', lambda: 'unknown')()
        
        get_logger().info(f"Step {step_num} Communication Status:")
        get_logger().info(f"  Agents: {len(gathered_results)}")
        get_logger().info(f"  Mode: {mode}")
        get_logger().info(f"  Emergency: {self._emergency_mode_enabled}")
        get_logger().info(f"  Total errors: {self._communication_errors}")
        get_logger().info(f"  Last distributed: {self._last_successful_gather}")

    def train_step_with_communication(self, batch_data, step_num, loss_fn=None, optimizer=None):
        """
        Enhanced training step with built-in communication.
        This method can be called directly from training loops.
        """
        
        try:
            # Forward pass (using existing GRPO logic)
            with torch.cuda.amp.autocast():
                outputs = self.forward(batch_data)
                loss = outputs.get('loss', None)
                
                if loss is None and loss_fn is not None:
                    loss = loss_fn(outputs, batch_data)
            
            # Prepare data for distributed gathering
            step_data = {
                'step': step_num,
                'loss': loss.item() if loss is not None else 0.0,
                'agent_id': self._robust_backend.get_id(),
                'batch_size': len(batch_data) if hasattr(batch_data, '__len__') else 1,
                'timestamp': time.time()
            }
            
            # Add model-specific outputs if available
            if 'logits' in outputs:
                step_data['has_logits'] = True
            if 'rewards' in outputs:
                step_data['avg_reward'] = torch.mean(outputs['rewards']).item()
            
            # DISTRIBUTED GATHERING
            gathered_results = self.robust_all_gather(step_data, step_num)
            
            # Process distributed results
            processed_outputs = self._process_distributed_results(gathered_results, outputs, step_num)
            
            # Backward pass
            if loss is not None:
                loss.backward()
                
                # Gradient clipping
                if hasattr(self, 'model'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Clear cache periodically
            if step_num % 10 == 0:
                self._smart_cache_clear()
            
            return processed_outputs
            
        except Exception as e:
            get_logger().error(f"Training step {step_num} failed: {e}")
            # Return minimal output to continue training
            return {
                'loss': torch.tensor(0.0) if 'loss' not in locals() else loss,
                'step': step_num,
                'error': str(e),
                'gathered_results': {self._robust_backend.get_id(): step_data} if 'step_data' in locals() else {}
            }

    def _process_distributed_results(self, gathered_results, original_outputs, step_num):
        """Process results from distributed gathering"""
        
        processed = original_outputs.copy() if isinstance(original_outputs, dict) else {}
        processed['step'] = step_num
        processed['num_agents'] = len(gathered_results)
        processed['gathered_results'] = gathered_results
        
        if len(gathered_results) == 1:
            # Single-node mode
            processed['training_mode'] = 'single_node'
            if step_num % 500 == 0:
                get_logger().info(f"Step {step_num}: Single-node training")
                
        else:
            # Multi-agent mode
            processed['training_mode'] = 'distributed'
            
            # Aggregate metrics from all agents
            losses = []
            batch_sizes = []
            
            for agent_id, data in gathered_results.items():
                if isinstance(data, dict):
                    if 'loss' in data and isinstance(data['loss'], (int, float)):
                        losses.append(data['loss'])
                    if 'batch_size' in data:
                        batch_sizes.append(data['batch_size'])
            
            if losses:
                processed['distributed_avg_loss'] = sum(losses) / len(losses)
                processed['distributed_loss_std'] = torch.std(torch.tensor(losses)).item()
            
            if batch_sizes:
                processed['total_batch_size'] = sum(batch_sizes)
            
            # Log distributed training info
            if step_num % 100 == 0:
                avg_loss = processed.get('distributed_avg_loss', 0.0)
                get_logger().info(f"Step {step_num}: Distributed training - {len(gathered_results)} agents, avg_loss = {avg_loss:.4f}")
        
        return processed

    # CRITICAL: Override all communication methods to prevent training stoppage
    def all_gather_object(self, obj):
        """Override parent's all_gather_object with robust version"""
        try:
            return self.robust_all_gather(obj, self._step_counter)
        except Exception as e:
            get_logger().error(f"all_gather_object failed: {e}")
            # Always return single-node result to prevent training stoppage
            return {self.get_id(): obj}

    def get_id(self):
        """Get agent ID - always returns valid ID"""
        try:
            if self._robust_backend:
                return self._robust_backend.get_id()
        except Exception as e:
            get_logger().warning(f"Failed to get ID from backend: {e}")
        
        # Fallback ID
        return f"trainer_{os.getpid()}"

    def set_communication_backend(self, backend):
        """Allow external setting of communication backend"""
        if backend is not None:
            self._robust_backend = create_robust_communication_wrapper(backend)
            get_logger().info("Communication backend updated with robust wrapper")

    # FRAMEWORK INTEGRATION: Override any communication-related methods
    def gather(self, *args, **kwargs):
        """Override any gather methods"""
        return self.all_gather_object(*args, **kwargs)

    def broadcast(self, obj, *args, **kwargs):
        """Override broadcast methods"""
        return self.all_gather_object(obj)

    def communicate(self, obj, *args, **kwargs):
        """Override generic communicate methods"""
        return self.all_gather_object(obj)

    # ORIGINAL METHODS (unchanged interface)
    
    def _is_model_quantized(self, model) -> bool:
        """Check if model is quantized"""
        # Check BnB attributes
        if hasattr(model, 'is_quantized') and model.is_quantized:
            return True
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            return True
        
        # Check quantization config
        if (hasattr(model, 'config') and 
            hasattr(model.config, 'quantization_config') and 
            model.config.quantization_config is not None):
            qconfig = model.config.quantization_config
            if hasattr(qconfig, 'load_in_4bit') and qconfig.load_in_4bit:
                return True
        
        # Check parameter dtypes (quantized models have int/uint params)
        int_params = 0
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()
            if 'int' in str(param.dtype).lower():
                int_params += param.numel()
        
        return total_params > 0 and int_params / total_params > 0.1

    def _reload_with_quantization(self, model, kwargs):
        """Reload model with 4-bit quantization"""
        model_name = getattr(model, 'name_or_path', kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct"))
        
        # Clear GPU memory
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Create quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Reload with quantization
        new_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        return new_model

    def _smart_cache_clear(self):
        """Clear cache only when memory usage is high"""
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            if allocated_gb > 3.5:
                torch.cuda.empty_cache()

    def _initialize_model(self, enable_gradient_checkpointing: bool = False):
        """
        Override to handle quantized models properly.
        """
        is_quantized = self._is_model_quantized(self.model)
        
        if is_quantized:
            pass  # For quantized models, don't cast dtype
        else:
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        """Original evaluate method unchanged"""
        if not self.judge_client:
            return
            
        try:
            model_name = self.model.name_or_path
        except AttributeError:
            model_name = "none"

        result = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name
        )
        
        if not result:
            return

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": result["question"]},
        ]
        input_ids = self.processing_class.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids, max_new_tokens=512)
        answer = self.processing_class.decode(outputs[0], skip_special_tokens=True)
        
        self._smart_cache_clear()
        
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )

    @torch.no_grad()
    def play_prg_game_logits(self, prg_history_dict: dict) -> dict:
        """Original PRG game method unchanged"""
        if not self.judge_client:
            return {'status': PRGGameStatus.ERROR}

        game_clue_dict = self.judge_client.get_current_clue()
        
        if not isinstance(game_clue_dict, dict):
            return {'status': PRGGameStatus.ERROR}
        
        game_id = game_clue_dict.get("game_id", -1)
        clue_id = game_clue_dict.get("clue_id", -1)
        rounds_remaining = game_clue_dict.get("rounds_remaining", -1)
        clue = game_clue_dict.get("clue") or ""
        choices = game_clue_dict.get("choices") or []
        
        if any(val < 0 for val in (game_id, clue_id, rounds_remaining)):
            return {'status': PRGGameStatus.NO_ACTIVE_GAME}
        if game_id in prg_history_dict and clue_id <= prg_history_dict[game_id]:
            return {'status': PRGGameStatus.ALREADY_ANSWERED}
        if not clue or not isinstance(choices, list) or not choices:
            return {'status': PRGGameStatus.ERROR}

        try:
            choices_str = ", ".join(choices)
            custom_prompt = f"{clue}\nPossible Answers: {choices_str}\nAnswer:"
            
            prompt = [
                {"role": "system", "content": PRG_SYSTEM_PROMPT_NO_THINKING},
                {"role": "user", "content": custom_prompt},
            ]
            input_ids = self.processing_class.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            input_ids = input_ids.to(self.model.device)
            choice_logits = self._get_choice_logits(input_ids, choices)
            choice_idx = torch.argmax(choice_logits).item()
            
            torch.cuda.empty_cache()
            
            return {
                "game_idx": game_id,
                "clue_idx": clue_id,
                "choice_idx": choice_idx,
                "choice": choices[choice_idx],
                "rounds_remaining": rounds_remaining,
                "status": PRGGameStatus.SUCCESS
            }

        except Exception as e:
            get_logger().info(f"Error while computing logits for choices: {e}")
            return {'status': PRGGameStatus.ERROR}

    def _get_choice_logits(self, input_ids: torch.Tensor, choices: List[str]) -> torch.Tensor:
        """Original choice logits method unchanged"""
        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        logits_list = []

        for choice in choices:
            answer_str = f"<answer>{choice}</answer>"
            choice_ids = self.processing_class(
                answer_str,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(device)

            seq = torch.cat([input_ids, choice_ids], dim=1)
            labels = seq.clone()
            labels[:, :prompt_len] = -100
            outputs = self.model(input_ids=seq, labels=labels)

            total_log_prob = -outputs.loss * choice_ids.size(1)
            logits_list.append(total_log_prob)

        return torch.stack(logits_list)

    def cleanup(self):
        """Clean shutdown of communication backend"""
        
        if self._robust_backend:
            try:
                if hasattr(self._robust_backend, 'backend'):
                    if hasattr(self._robust_backend.backend, 'shutdown'):
                        self._robust_backend.backend.shutdown()
                elif hasattr(self._robust_backend, 'shutdown'):
                    self._robust_backend.shutdown()
                    
            except Exception as e:
                get_logger().warning(f"Communication cleanup error: {e}")

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass


# Simple usage example
def example_usage():
    """Example showing how to use the enhanced trainer"""
    
    # Initialize trainer (same as before)
    trainer = GRPOTrainerModule([], model_id="Qwen/Qwen2.5-3B-Instruct")
    
    # Training loop with robust communication
    for step in range(100):
        try:
            # Mock batch data
            batch_data = {'input_ids': torch.randint(0, 1000, (2, 10))}
            
            # OPTION 1: Enhanced training step (recommended)
            outputs = trainer.train_step_with_communication(batch_data, step)
            
            # OPTION 2: Manual gathering 
            # outputs = trainer(batch_data)
            # gathered = trainer.robust_all_gather({'loss': outputs['loss'].item()}, step)
            
            if step % 20 == 0:
                mode = outputs.get('training_mode', 'unknown')
                agents = outputs.get('num_agents', 1)
                get_logger().info(f"Step {step}: {mode} mode with {agents} agents")
                
        except Exception as e:
            get_logger().error(f"Step {step} failed: {e}")
            continue
    
    # Cleanup
    trainer.cleanup()
    get_logger().info("Training completed successfully")


if __name__ == "__main__":
    example_usage()
