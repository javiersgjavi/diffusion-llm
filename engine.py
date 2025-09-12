import logging
import torch
from pytorch_lightning import LightningModule
from rich.live import Live
from rich.console import Console
from utils import MaskGenerator, ModelOutput, SeedController, RandomRemaskStrategy, BanSpecialTokens, GreedySampling, MultinomialSampling, build_optimizer_and_scheduler
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
    
class LLADAEngine(LightningModule):
    def __init__(
        self,
        t_steps: int = 512,
        total_steps: int | None = None,
    ) -> None:
        super().__init__()

        # Model and Tokenizer
        model_name = "distilbert-base-uncased"
        self.model: DistilBertForMaskedLM = DistilBertForMaskedLM.from_pretrained(model_name)
        self.tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(model_name)

        self.special_token_ids = {
            'pad_token_id': self.tokenizer.pad_token_id,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'mask_token_id': self.tokenizer.mask_token_id,
        }
        # Helper classes
        self.mask_generator = MaskGenerator(
            mask_token_id=self.special_token_ids['mask_token_id'],
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
                'input_ids': x.clone(),
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

    def generate(
            self,
            t_steps: int = 1024,
            n_tokens: int | None = None,
            ban_special: bool = True,
            sampling: str = "multinomial",
            ) -> None:

        logging.info("Generating sample text...")

        # Prepare model and variables
        self.model.eval()

        n_tokens = n_tokens if n_tokens is not None else self.max_tokens
        t_values_sampling = reversed(torch.linspace(0, 1, t_steps +1)[1:])

        special_tokens_to_avoid = ['cls_token_id', 'mask_token_id']

        cleaner = BanSpecialTokens(
            banned_token_ids=[self.special_token_ids[t] for t in special_tokens_to_avoid]
        )

        if sampling == "greedy":
            sampling_strategy = GreedySampling()
        elif sampling == "multinomial":
            sampling_strategy = MultinomialSampling()
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling}")

        # Start from all masked tokens
        mask = torch.ones(
            size=(1, n_tokens),
            device=self.device,
            dtype=torch.int64
        )

        x = mask.clone() * self.special_token_ids['mask_token_id']

        # Ensure the first and last tokens are CLS
        x[:,0] = self.special_token_ids['cls_token_id']
        mask[:,0] = 0

        # Ensure the last token is SEP
        x[:, -1] = self.special_token_ids['sep_token_id']
        mask[:, -1] = 0

        remask_strategy = RandomRemaskStrategy(
            self.mask_generator.mask_token_id,
            t_min=t_values_sampling[-1],
            mask=mask,
        )

        # Generate tokens step by step
        console = Console()
        with torch.no_grad():
            with Live("", refresh_per_second=20, console=console) as live:

                for t in t_values_sampling:
                    t = t.to(self.device)

                    outputs: ModelOutput = self.forward(x, t, need_mask=False)
                    
                    if ban_special:
                        outputs = cleaner(outputs)

                    outputs = sampling_strategy(outputs)

                    # If t is not the minimum t, we apply the remasking strategy
                    if not t == t_values_sampling[-1]:
                        x = remask_strategy(x, outputs, t)

                    else:
                        x = outputs.tokens

                    live.update(self.decode_tokens(x, skip_special_tokens=False))

                    if not remask_strategy.has_masked_tokens():
                        break

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
        return torch.optim.AdamW(self.parameters(), lr=4e-4, weight_decay=0.01)
        #return build_optimizer_and_scheduler(
        #    self.parameters(),
        #    total_steps=self.total_steps,
        #)
    
    def on_train_start(self):
        self.seed_controller.set_train_seed()
    
    def on_validation_start(self):
        self.seed_controller.set_val_seed()
        self.generate()
    