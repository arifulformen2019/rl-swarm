from typing import Any, Optional, List

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
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.

        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Patch: load the model with 4-bit quantization if not already provided
        if not models:
            model_id = kwargs.get("model_id", "Qwen2.5-3B-Instruct-bnb-4bit")
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

    def _initialize_model(self, enable_gradient_checkpointing: bool = False):
        """
        Override to handle quantized models properly.
        Quantized models cannot be cast to different dtypes.
        """
        # Check if model is quantized (bitsandbytes)
        is_quantized = (
            hasattr(self.model, 'is_quantized') and self.model.is_quantized
        ) or (
            hasattr(self.model, 'config') and 
            hasattr(self.model.config, 'quantization_config') and 
            self.model.config.quantization_config is not None
        )
        
        if is_quantized:
            # For quantized models, don't cast dtype - just ensure it's on the right device
            get_logger().info("Model is quantized, skipping dtype casting")
            # Model should already be on correct device due to device_map="auto"
        else:
            # For regular models, apply the normal dtype casting
            self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        # Enable gradient checkpointing if requested
        if enable_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        if not self.judge_client:
            return
            
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

        # TODO: Make the dtype changes from genrl here?
        input_ids = input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids, max_new_tokens=512)
        answer = self.processing_class.decode(
            outputs[0], skip_special_tokens=True
        )
        
        # Submit answer to judge service
        self.judge_client.submit_answer(
            session_id=result["session_id"],
            round_number=state.round,
            user_answer=answer
        )

    @torch.no_grad()
    def play_prg_game_logits(
        self, prg_history_dict: dict
    ) -> dict:
        if not self.judge_client:
            return {'status': PRGGameStatus.ERROR}

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

            # TODO: Make the dtype changes from genrl here?
            input_ids = input_ids.to(self.model.device)
            
            # Get logits for each choice
            choice_logits = self._get_choice_logits(input_ids, choices)
            
            # Select the choice with highest probability
            choice_idx = torch.argmax(choice_logits).item()
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
            # TODO: Make the dtype changes from genrl here?
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

        # stack into a single tensor of shape (num_choices,)
        return torch.stack(logits_list)
