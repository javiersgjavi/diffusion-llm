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

        x_tokens[mask] = self.mask_token_id
        x['input_ids'] = x_tokens
        return x, mask
    
@dataclass
class ModelOutput:
    logits: torch.Tensor
    mask: torch.Tensor
    tokens: torch.Tensor

class GreedySampling:
    """ Greedy sampling strategy: select the token with the highest probability at each step.  """
    def __call__(self, output: ModelOutput) -> torch.Tensor:
        output.tokens = output.logits.argmax(dim=-1)
        return output
    
class MultinomialSampling:
    """ Multinomial sampling strategy: sample tokens based on their probabilities."""
    def __call__(self, output: ModelOutput) -> torch.Tensor:
        probs = torch.nn.functional.softmax(output.logits, dim=-1)
        for batch_idx in range(probs.shape[0]):
            output.tokens[batch_idx] = torch.multinomial(probs[batch_idx], num_samples=1).squeeze(-1)
        return output
    
class BanSpecialTokens:
    """ Class to ban special tokens during generation by setting their logits to -inf. """
    def __init__(self, banned_token_ids: list[int]):
        self.banned_token_ids = torch.tensor(list(banned_token_ids), dtype=torch.long)

    def _check_same_device(self, output: ModelOutput) -> None:
        if self.banned_token_ids.device != output.logits.device:
            self.banned_token_ids = self.banned_token_ids.to(output.logits.device)

    def __call__(self, output: ModelOutput) -> torch.Tensor:
        self._check_same_device(output)
        output.logits[..., self.banned_token_ids] = float('-inf')
        return output

class RandomRemaskStrategy:
    """ Class to implement a random remasking strategy during training.  """
    def __init__(self, mask_token_id: int, t_min: torch.tensor, mask: torch.Tensor | None = None):
        self.mask_token_id = mask_token_id
        self.t_min = t_min
        self.mask = mask

    def _check_initial_mask(self, outputs: ModelOutput) -> bool:
        if self.mask is None:
            self.mask = torch.ones_like(outputs.tokens, device=outputs.tokens.device, dtype=torch.bool)

        if self.mask.device != outputs.tokens.device:
            self.mask = self.mask.to(outputs.tokens.device)

        if self.mask.dtype != torch.bool:
            self.mask = self.mask.to(torch.bool)

    def has_masked_tokens(self) -> bool:
        return torch.sum(self.mask) > 0

    def __call__(self, x: torch.Tensor, outputs: ModelOutput, t: torch.Tensor) -> torch.Tensor:
        self._check_initial_mask(outputs)

        s = t - self.t_min
        new_mask = torch.rand(outputs.tokens.shape, device=outputs.tokens.device) < s
        
        tokens_new = x.clone()
        tokens_new[self.mask] = outputs.tokens[self.mask]


        self.mask = self.mask & new_mask
        tokens_new[self.mask] = self.mask_token_id
       
        #outputs.tokens[self.mask] = self.mask_token_id
        return tokens_new
    
def build_optimizer_and_scheduler(
    params,
    total_steps: int,
    lr_peak: float = 4e-4,
    weight_decay: float = 0.01,
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
