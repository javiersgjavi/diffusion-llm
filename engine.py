import logging
import torch
from pytorch_lightning import LightningModule
from rich.live import Live
from rich.console import Console
from utils import MaskGenerator, ModelOutput, SeedController, RandomRemaskStrategy, build_optimizer_and_scheduler
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
    
class LLADAEngine(LightningModule):
    def __init__(
        self,
        t_steps: int = 1024,
        total_steps: int | None = None,
    ) -> None:
        super().__init__()

        # Model and Tokenizer
        model_name = "distilbert-base-uncased"
        self.model: DistilBertForMaskedLM = DistilBertForMaskedLM.from_pretrained(model_name)
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Helper classes
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.mask_generator = MaskGenerator(
            mask_token_id=self.tokenizer.mask_token_id,
            special_token_ids=self.special_token_ids,
        )
        self.seed_controller = SeedController()

        # other attributes
        self.max_tokens = self.tokenizer.model_max_length
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.t_values_sampling = reversed(torch.linspace(0, 1, t_steps +1)[1:])
        self.total_steps = total_steps # For optimizer scheduler

    # ----------- Main methods for training, validation and generation -----------

    def forward(
            self,
            x: dict | torch.Tensor,
            t: torch.Tensor,
            need_mask: bool = True
        ) -> ModelOutput:
        """ Implement the forward pass of the model and the masking of the input tokens if needed """
        
        if isinstance(x, torch.Tensor):
            x = {
                'input_ids': x,
                'attention_mask': torch.ones_like(x)
            }
            
        if need_mask:
            x_masked, mask = self.mask_generator(x, t)
        else:
            x_masked, mask = x, None

        outputs: torch.Tensor = self.model(**x_masked)['logits']
        tokens = outputs.argmax(dim=-1)

        return ModelOutput(logits=outputs, mask=mask, tokens=tokens)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:

        # Prepare inputs
        x_tokenized: dict[str, torch.Tensor] = self.tokenize_batch(batch)
        target: torch.Tensor = x_tokenized['input_ids'].clone()
        tensor_shape = target.shape

        # Sample t and expand to the shape of the input tokens
        t = torch.rand(tensor_shape[0], device=self.device)
        t = t.reshape(-1, 1).expand(-1, tensor_shape[1])

        # Forward pass
        outputs: ModelOutput = self.forward(x_tokenized, t)
        loss = self.loss_fn(outputs.logits, target, outputs.mask, t)
        self.log_loss('train_loss', loss, outputs)

        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        # Prepare inputs
        x_tokenized = self.tokenize_batch(batch)

        # Accumulate loss over different t values
        num_losses = 0
        cum_loss = torch.tensor([0.0], device=self.device)

        for t in self.t_values_sampling:
            
            # Sample t and expand to the shape of the input tokens
            t = t.reshape(-1, 1).expand(x_tokenized['input_ids'].shape)
            t = t.to(self.device)

            # Forward pass
            outputs: ModelOutput = self.forward(x_tokenized, t)
            loss = self.loss_fn(outputs.logits, x_tokenized['input_ids'], outputs.mask, t)

            # Compute loss. Some times the mask can be all false, so the loss is nan
            if not torch.isnan(loss):
                cum_loss += loss
                num_losses += 1

        loss /= num_losses

        self.log_loss('val_loss', cum_loss, outputs)

    def generate(self, t_steps: int = 1024) -> None:

        logging.info("Generating sample text...")
        # Prepare model and variables
        self.model.eval()
        t_values_sampling = reversed(torch.linspace(0, 1, t_steps +1)[1:])

        remask_strategy = RandomRemaskStrategy(
            self.mask_generator.mask_token_id,
            t_min=t_values_sampling[-1]
        )

        # Start from all masked tokens
        x = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=(1, self.max_tokens//2),
            device=self.device
        )

        mask = torch.ones(
            size=(1, self.max_tokens//2),
            device=self.device,
            dtype=torch.int64
        )

        x = mask.clone() * self.mask_generator.mask_token_id

        # Generate tokens step by step
        console = Console()
        with torch.no_grad():
            with Live("", refresh_per_second=20, console=console) as live:

                for t in t_values_sampling:
                    t = t.to(self.device)

                    outputs: ModelOutput = self.forward(x, t, need_mask=False)
                    outputs.mask = mask.bool()

                    # Avoid generating special tokens
                    logits = self._ban_tokens_in_logits(outputs.logits, self.special_token_ids)
                    tokens = logits.argmax(dim=-1)
                    outputs.tokens = tokens

                    # If t is not the minimum t, we apply the remasking strategy
                    if not t == t_values_sampling[-1]:
                        x = remask_strategy(outputs, t)

                    else:
                        x = outputs.tokens

                    live.update(self.decode_tokens(x, skip_special_tokens=True))

    def loss_fn(self, outputs, labels, mask, t) -> torch.Tensor:
        """ Compute loss following the equation 5 of the paper """
        loss = self.criterion(outputs[mask], labels[mask])
        t = torch.max(t, torch.tensor(1e-5, device=self.device))
        loss: torch.Tensor = 1/t[mask] * loss
        return loss.mean()
    
    def decode_tokens(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """ Decode tokens to string """
        return self.tokenizer.decode(tokens[0], skip_special_tokens=skip_special_tokens)
    
    def _ban_tokens_in_logits(self, logits: torch.Tensor, banned_ids: set[int]) -> torch.Tensor:
        """Ban tokens in logits putting -inf"""
        if not banned_ids:
            return logits
        banned = torch.tensor(list(banned_ids), device=logits.device, dtype=torch.long)
        logits[..., banned] = -float('inf')
        return logits

    
    def log_loss(self, name: str, loss: torch.Tensor, outputs: ModelOutput) -> None:
        """ Log loss to TensorBoard and progress bar """
        self.log(
            name,
            loss.detach(),
            prog_bar=True,
            on_epoch=True,
            logger=True,
            batch_size=outputs.mask.shape[0],
            sync_dist=True,
        )

    def tokenize_batch(self, batch: dict) -> dict:
        x = batch['text']
        x_tokenized = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        x_tokenized = x_tokenized.to(self.device)
        return x_tokenized
        
    # ----------- Pytorch Lightning specific methods -----------

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            self.parameters(),
            total_steps=self.total_steps,
        )
    
    def on_train_start(self):
        self.seed_controller.set_train_seed()
    
    def on_validation_start(self):
        self.seed_controller.set_val_seed()
        self.generate()
    