from typing import Any, Optional, List
import gc
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
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    With 4-bit quantization support and aggressive memory management.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module with 4-bit quantization support.

        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Memory management settings
        self.memory_cleanup_frequency = kwargs.get("memory_cleanup_frequency", 1)  # Clean every step
        self.max_memory_gb = kwargs.get("max_memory_gb", 6.0)  # Target max VRAM usage
        self.step_counter = 0
        self.enable_aggressive_cleanup = kwargs.get("enable_aggressive_cleanup", True)
        
        # Track initial memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            get_logger().info(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Check and fix quantization for existing models
        if models:
            for i, model in enumerate(models):
                is_quantized = self._is_model_quantized(model)
                
                if not is_quantized:
                    get_logger().warning(f"Model {i} is not quantized, reloading with 4-bit quantization...")
                    try:
                        models[i] = self._reload_with_quantization(model, kwargs)
                        get_logger().info(f"Model {i} successfully quantized to 4-bit")
                    except Exception as e:
                        get_logger().error(f"Failed to reload model {i} with quantization: {e}")

        # Fallback: load model with quantization if no models provided
        if not models:
            model_id = kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
            get_logger().info(f"Loading model {model_id} with 4-bit quantization...")
            
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
            get_logger().info(f"Model loaded successfully with 4-bit quantization")
        
        super().__init__(models, **kwargs)
        judge_base_url = kwargs.get("judge_base_url", None)
        self.judge_client = JudgeClient(judge_base_url) if judge_base_url else None
        
        # Force initial cleanup
        self._aggressive_memory_cleanup()

    def _is_model_quantized(self, model) -> bool:
        """
        Check if model is quantized.
        
        Args:
            model: The model to check
            
        Returns:
            bool: True if model is quantized, False otherwise
        """
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
        """
        Reload model with 4-bit quantization.
        
        Args:
            model: The model to reload
            kwargs: Configuration arguments
            
        Returns:
            The quantized model
        """
        model_name = getattr(model, 'name_or_path', kwargs.get("model_id", "Qwen/Qwen2.5-3B-Instruct"))
        
        get_logger().info(f"Reloading model {model_name} with 4-bit quantization...")
        
        # Clear GPU memory
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
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
        
        get_logger().info(f"Model {model_name} reloaded with 4-bit quantization")
        return new_model

    def _get_memory_stats(self):
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }

    def _log_memory_stats(self, stage=""):
        """Log memory statistics"""
        stats = self._get_memory_stats()
        if stats:
            get_logger().info(
                f"[{stage}] GPU Memory - "
                f"Allocated: {stats['allocated']:.2f}GB, "
                f"Reserved: {stats['reserved']:.2f}GB, "
                f"Peak: {stats['max_allocated']:.2f}GB"
            )

    def _aggressive_memory_cleanup(self):
        """Aggressively clean up GPU memory"""
        if not torch.cuda.is_available():
            return
        
        # Clear gradients
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None
        
        # Python garbage collection
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Synchronize to ensure cleanup is complete
        torch.cuda.synchronize()

    def _smart_cache_clear(self):
        """Clear cache based on memory usage"""
        if not torch.cuda.is_available():
            return
            
        stats = self._get_memory_stats()
        if stats is None:
            return
        
        # Always clean if above threshold
        if stats['allocated'] > self.max_memory_gb:
            get_logger().warning(
                f"Memory usage high ({stats['allocated']:.2f}GB > {self.max_memory_gb}GB), "
                f"performing aggressive cleanup..."
            )
            self._aggressive_memory_cleanup()
            
            # Log after cleanup
            new_stats = self._get_memory_stats()
            if new_stats:
                get_logger().info(
                    f"After cleanup: {new_stats['allocated']:.2f}GB "
                    f"(freed {stats['allocated'] - new_stats['allocated']:.2f}GB)"
                )
        
        # Periodic cleanup based on frequency
        elif self.enable_aggressive_cleanup and self.step_counter % self.memory_cleanup_frequency == 0:
            self._aggressive_memory_cleanup()

    def _initialize_model(self, enable_gradient_checkpointing: bool = False):
        """
        Override to handle quantized models properly.
        Quantized models should not have dtype cast applied.
        
        Args:
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
        """
        is_quantized = self._is_model_quantized(self.model)
        
        if is_quantized:
            get_logger().info("Model is quantized - skipping dtype casting")
            # For quantized models, don't cast dtype
            pass
        else:
            get_logger().info(f"Moving model to device={self.device}, dtype={self.dtype}")
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            get_logger().info("Gradient checkpointing enabled")

    def train_step(self, *args, **kwargs):
        """
        Override train_step to add memory cleanup
        """
        self.step_counter += 1
        
        # Log memory before training
        if self.step_counter % 100 == 0:
            self._log_memory_stats(f"Before Step {self.step_counter}")
        
        # Call parent train_step
        result = super().train_step(*args, **kwargs)
        
        # Clean memory after step
        self._smart_cache_clear()
        
        # Log memory after training
        if self.step_counter % 100 == 0:
            self._log_memory_stats(f"After Step {self.step_counter}")
        
        return result

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        if not self.judge_client:
            return
        
        # Log memory before evaluation
        self._log_memory_stats("Before Evaluate")
            
        try:
            model_name = self.model.name_or_path
        except AttributeError:
            model_name = "none"

        # Request question from judge service
        result = self.judge_client.request_question(
            user_id=state.peer_id,
            round_number=state.round,
            model_name=model_name
        )
        
        if not result:
            return

        # Generate answer using the model
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
        answer = self.processing_class.decode(
            outputs[0], skip_special_tokens=True
        )
        
        # Clean up immediately after generation
        del input_ids, outputs
        self._aggressive_memory_cleanup()
        
        # Submit answer to judge service
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )
        
        # Log memory after evaluation
        self._log_memory_stats("After Evaluate")

    @torch.no_grad()
    def play_prg_game_logits(
        self, prg_history_dict: dict
    ) -> dict:
        if not self.judge_client:
            return {'status': PRGGameStatus.ERROR}

        # Log memory before PRG game
        self._log_memory_stats("Before PRG Game")

        # Get current clue from judge service
        game_clue_dict = self.judge_client.get_current_clue()
        
        if not isinstance(game_clue_dict, dict):
            return {'status': PRGGameStatus.ERROR}
        
        # If no clue or game_id or clue_id is -1, take no action
        game_id = game_clue_dict.get("game_id", -1)
        clue_id = game_clue_dict.get("clue_id", -1)
        rounds_remaining = game_clue_dict.get("rounds_remaining", -1)
        clue = game_clue_dict.get("clue") or ""
        choices = game_clue_dict.get("choices") or []
        
        # No active game
        if any(val < 0 for val in (game_id, clue_id, rounds_remaining)):
            return {'status': PRGGameStatus.NO_ACTIVE_GAME}
        # We have already answered this clue
        if game_id in prg_history_dict and clue_id <= prg_history_dict[game_id]:
            return {'status': PRGGameStatus.ALREADY_ANSWERED}
        # malformed input
        if not clue or not isinstance(choices, list) or not choices:
            return {'status': PRGGameStatus.ERROR}
        
        get_logger().info(f"New clue received for PRG: {game_clue_dict}")

        try:
            choices_str = ", ".join(choices)
            custom_prompt = f"{clue}\nPossible Answers: {choices_str}\nAnswer:"
            
            # Generate answer using the model with custom prompt
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
            
            # Get logits for each choice
            choice_logits = self._get_choice_logits(input_ids, choices)
            
            # Select the choice with highest probability
            choice_idx = torch.argmax(choice_logits).item()
            
            # Clean up immediately after computation
            del input_ids, choice_logits
            self._aggressive_memory_cleanup()
            
            # Log memory after PRG game
            self._log_memory_stats("After PRG Game")
            
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
            # Clean up on error too
            self._aggressive_memory_cleanup()
            return {'status': PRGGameStatus.ERROR}

    def _get_choice_logits(self, input_ids: torch.Tensor, choices: List[str]) -> torch.Tensor:
        """
        Returns a tensor of shape (len(choices),) giving, for each choice,
        the sum of log-probabilities that the model assigns to generating
        "<answer>{choice}</answer>" after the given input_ids.
        """

        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        logits_list = []

        for choice in choices:
            # 1) build the full token sequence: prompt + "<answer>…</answer>"
            answer_str = f"<answer>{choice}</answer>"
            choice_ids = self.processing_class(
                answer_str,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(device)    # shape (1, L)

            seq = torch.cat([input_ids, choice_ids], dim=1)  # (1, prompt_len + L)

            # build labels that only include the answer positions
            labels = seq.clone()
            labels[:, :prompt_len] = -100  # ignore prompt positions in loss
            outputs = self.model(input_ids=seq, labels=labels)
            # outputs.loss is average negative log-likelihood over the L answer tokens

            total_log_prob = -outputs.loss * choice_ids.size(1)
            logits_list.append(total_log_prob)
            
            # Clean up intermediate tensors
            del choice_ids, seq, labels, outputs

        # stack into a single tensor of shape (num_choices,)
        result = torch.stack(logits_list)
        
        # Clean up list
        del logits_list
        
        return result

    def cleanup(self):
        """Cleanup method for proper shutdown"""
        get_logger().info("Cleaning up trainer resources...")
        
        # Final memory stats
        self._log_memory_stats("Final Cleanup")
        
        # Aggressive cleanup
        self._aggressive_memory_cleanup()
        
        get_logger().info("Trainer cleanup complete")

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
