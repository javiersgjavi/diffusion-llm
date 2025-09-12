import torch
from engine import LLADAEngine

def main():
    torch.set_float32_matmul_precision('medium')
    # Cargar el checkpoint pasando los argumentos requeridos
    engine = LLADAEngine.load_from_checkpoint(
        #"logs_2/lightning_logs/version_22/checkpoints/epoch=2-step=49682.ckpt",
        #"logs/lightning_logs/version_8/checkpoints/epoch=2-step=85778.ckpt",
        "logs_2/lightning_logs/version_25/checkpoints/epoch=2-step=36434.ckpt",
    )

    engine.generate(t_steps=512, n_tokens=256)

if __name__ == "__main__":
    main()