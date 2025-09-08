from engine import LLADAEngine

def main():
    # Cargar el checkpoint pasando los argumentos requeridos
    engine = LLADAEngine.load_from_checkpoint(
        "logs/lightning_logs/version_8/checkpoints/epoch=2-step=85778.ckpt",
    )

    engine.generate()

if __name__ == "__main__":
    main()