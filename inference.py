from engine import LLADAEngine

def main():
    # Cargar el checkpoint pasando los argumentos requeridos
    engine = LLADAEngine.load_from_checkpoint(
        "logs_2/lightning_logs/version_14/checkpoints/epoch=2-step=45167.ckpt",
    )

    engine.generate()

if __name__ == "__main__":
    main()