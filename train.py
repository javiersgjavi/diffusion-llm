import torch
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from engine import LLADAEngine

def get_config_gpu():
    res = {}

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus == 0:
        res['devices'] = None
        res['strategy'] = None
        res['precision'] = 32
        
    elif num_gpus == 1:
        torch.set_float32_matmul_precision('medium')
        res['devices'] = 1
        res['strategy'] = None
        res['precision'] = "16-mixed"

    else:
        torch.set_float32_matmul_precision('medium')
        res['devices'] = -1
        res['strategy'] = DDPStrategy(find_unused_parameters=False)
        res['precision'] = "16-mixed"

    return res

def main():

    seed_everything(42)

    max_epochs = 3
    batch_size = 64
    desired_batch_size = 64  # Effective batch size

    dataset = load_dataset("roneneldan/TinyStories")

    dataset_train = dataset['train']
    dataset_validation = dataset['validation'].shuffle(seed=42).select(range(128))

    accumulate_grad_batches = desired_batch_size // batch_size
    
    logging.basicConfig(level=logging.INFO)

    logging.info(f'Batch size={batch_size}')
    logging.info(f'Desired batch size={desired_batch_size}')
    logging.info(f'Using accumulate grad batches={accumulate_grad_batches}')

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size, num_workers=4)

    # Compute total_steps: steps_per_epoch * epochs. Use desired global batch size.
    dataset_size = len(dataset_train)
    steps_per_epoch = dataset_size // desired_batch_size
    total_steps = steps_per_epoch * max_epochs
    
    logging.info(f'Dataset size={dataset_size}')
    logging.info(f'Steps per epoch={steps_per_epoch}')
    logging.info(f'Total steps={total_steps}')

    engine = LLADAEngine(total_steps=total_steps)

    config_gpu = get_config_gpu()
    logging.info(f'GPU config: {config_gpu}')
    
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir="logs"),
        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min')],
        enable_progress_bar=True,
        val_check_interval=0.2,
        accumulate_grad_batches=accumulate_grad_batches,
        #gradient_clip_val=1.0,
        accelerator="auto",
        devices=config_gpu['devices'],
        precision=config_gpu['precision'],
        log_every_n_steps=200,
        )
    
    trainer.fit(engine, dataloader_train, dataloader_validation)

    engine.generate()
    
if __name__ == "__main__":
    main()