import torch
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from engine import LLADAEngine
    
def main():

    seed_everything(42)

    max_epochs = 3
    batch_size = 8
    desired_batch_size = 256

    dataset = load_dataset("roneneldan/TinyStories")

    dataset_train = dataset['train']
    dataset_validation = dataset['validation'].shuffle(seed=42).select(range(32))

    accumulate_grad_batches = desired_batch_size // batch_size
    
    logging.basicConfig(level=logging.INFO)

    logging.info(f'Batch size={batch_size}')
    logging.info(f'Desired batch size={desired_batch_size}')
    logging.info(f'Using accumulate grad batches={accumulate_grad_batches}')

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size)

    # Compute total_steps: steps_per_epoch * epochs. Use desired global batch size.
    dataset_size = len(dataset_train)
    steps_per_epoch = (dataset_size + desired_batch_size - 1) // desired_batch_size
    total_steps = steps_per_epoch * max_epochs
    
    logging.info(f'Dataset size={dataset_size}')
    logging.info(f'Steps per epoch={steps_per_epoch}')
    logging.info(f'Total steps={total_steps}')

    engine = LLADAEngine(total_steps=total_steps)

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir="logs"),
        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min')],
        enable_progress_bar=True,
        val_check_interval=0.01,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        precision=16 if torch.cuda.is_available() else 32,
        )
    
    trainer.fit(engine, dataloader_train, dataloader_validation)

    engine.generate()
    
if __name__ == "__main__":
    main()