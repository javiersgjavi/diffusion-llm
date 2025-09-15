# LLaDA: Large Language Diffusion Models

A PyTorch Lightning implementation of **LLaDA** (Large Language Diffusion with mAsking) - a novel approach that challenges the autoregressive paradigm in large language models.

![Generation Demo](generation.gif)

## Why this project?

After working extensively with various text diffusion models that failed to deliver satisfactory results, I was excited to implement the LLaDA approach when the paper was released. This implementation represents my successful attempt to get text diffusion models working effectively.

## How does LLaDA work?

Unlike autoregressive models that generate token by token from left to right, LLaDA works very differently:

1. **Starts with completely masked text** (all tokens are [MASK])
2. **Iteratively unmask tokens** using a reverse diffusion process
3. **Doesn't start from Gaussian noise** like traditional diffusions, but from masked tokens
4. **Can generate in any direction** thanks to bidirectional dependencies

It's like having hidden text and gradually revealing words until you have the complete text.

## Key features

- ✅ **Masked token diffusion** instead of Gaussian noise
- ✅ **Bidirectional generation** - not limited to left-to-right
- ✅ **Two sampling strategies**: greedy (deterministic) and multinomial (probabilistic)
- ✅ **PyTorch Lightning training** - easy to use and scale
- ✅ **Automatic multi-GPU support**

## Quick installation

```bash
git clone https://github.com/yourusername/diffusion-llm.git
cd diffusion-llm
uv sync
```

## Basic usage

### Train the model
```bash
uv run train.py
```

### Generate text
```bash
# With greedy sampling (deterministic)
uv run inference.py --sampling greedy --n_tokens 50

# With multinomial sampling (more diverse)
uv run inference.py --sampling multinomial --n_tokens 50
```

### From Python
```python
from engine import LLADAEngine

# Load trained model
engine = LLADAEngine.load_from_checkpoint("weights/model.ckpt")

# Generate text
engine.generate(sampling="multinomial", n_tokens=50)
```

## Architecture

The model is built on top of DistilBERT and implements:

- **MaskGenerator**: Masks tokens during training
- **RandomRemaskStrategy**: Implements the reverse diffusion process
- **Sampling strategies**: GreedySampling and MultinomialSampling
- **Custom loss**: Optimization following equation 5 from the paper

## Why is this important?

LLaDA demonstrates that **you don't need to generate left-to-right** to have an effective language model. After struggling with other text diffusion approaches, this implementation finally shows that effective text diffusion is achievable.

The original paper shows that LLaDA 8B is competitive with LLaMA3 8B on many tasks, and even outperforms GPT-4o on reverse reasoning tasks (like completing poems backwards).

## References

- [Original LLaDA paper](https://arxiv.org/html/2502.09992v2)
- [Official implementation](https://ml-gsai.github.io/LLaDA-demo/)

## License

MIT License - you can use this code freely.

---

*Note: This implementation focuses on the generation component of LLaDA. For the complete system including supervised fine-tuning, consult the original paper.*