import logging
import torch
from dataclasses import dataclass
import copy
from transformers import get_cosine_schedule_with_warmup

@dataclass
class SeedController:
    """
    Class to control random seed for reproducibility between training and validation.
    """
    val_seed: torch.Tensor = torch.get_rng_state()
    train_seed: torch.Tensor = torch.get_rng_state()

    def set_train_seed(self) -> None:
        torch.set_rng_state(self.train_seed)

    def set_val_seed(self) -> None:
        self.train_seed = torch.get_rng_state()
        torch.set_rng_state(self.val_seed)

@dataclass
class MaskGenerator:
    """
    Class to generate masks for input tokens based on a given probability t.
    """
    mask_token_id: int = 103
    special_token_ids: set[int] | None = None

    def _generate_mask(self, x: torch.Tensor, t: float) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) < t
        return mask
    
    def __call__(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Generate a mask for the input tokens and replace masked tokens with the mask token ID.
        """
        x = copy.deepcopy(x)
        x_tokens = x['input_ids']
        mask = self._generate_mask(x_tokens, t)

        # Avoid masking special tokens (CLS, SEP, PAD, MASK, etc.)
        if self.special_token_ids is not None and len(self.special_token_ids) > 0:
            special_ids_tensor = torch.tensor(list(self.special_token_ids), device=x_tokens.device, dtype=x_tokens.dtype)
            special_positions = torch.isin(x_tokens, special_ids_tensor)
            mask = mask & ~special_positions

        x_tokens[mask] = self.mask_token_id
        x['input_ids'] = x_tokens
        return x, mask
    
@dataclass
class ModelOutput:
    logits: torch.Tensor
    mask: torch.Tensor
    tokens: torch.Tensor

class RandomRemaskStrategy:
    """
    Class to implement a random remasking strategy during training.
    """
    def __init__(self, mask_token_id: int, t_min: torch.tensor):
        self.mask_token_id = mask_token_id
        self.t_min = t_min
        self.mask = None

    def _check_initial_mask(self, outputs: ModelOutput) -> bool:
        if self.mask is None:
            self.mask = torch.ones_like(outputs.tokens, device=outputs.tokens.device, dtype=torch.bool)

    def __call__(self, outputs: ModelOutput, t: torch.Tensor) -> torch.Tensor:
        self._check_initial_mask(outputs)

        s = t - self.t_min
        new_mask = torch.rand(outputs.tokens.shape, device=outputs.tokens.device) < s
        
        self.mask = self.mask & new_mask
        outputs.tokens[self.mask] = self.mask_token_id
        return outputs.tokens
        
def build_optimizer_and_scheduler(
    params,
    total_steps: int,
    lr_peak: float = 4e-4,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.05,
):
    """
    Builds AdamW optimizer and cosine scheduler with warmup using
    `transformers.get_cosine_schedule_with_warmup`.
    """
    
    logger = logging.getLogger(__name__)
    optimizer = torch.optim.AdamW(params, lr=lr_peak, weight_decay=weight_decay)

    # Calculate warmup steps based on ratio
    num_warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    logger.info(f"Scheduler created: total_steps={total_steps}, num_warmup_steps={num_warmup_steps}, warmup_ratio={warmup_ratio}")
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
    }
