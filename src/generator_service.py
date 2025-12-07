import os
import numpy as np
import torch
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import Generator
from config import (
    DEVICE,
    LATENT_DIM,
    GENERATOR_WEIGHTS_PATH,
    GPT2_ADAPTER_DIR,
)

def load_generator() -> tuple[Generator, bool]:
    gen = Generator().to(DEVICE)
    ok = False
    if GENERATOR_WEIGHTS_PATH.exists():
        state = torch.load(GENERATOR_WEIGHTS_PATH, map_location=DEVICE)
        gen.load_state_dict(state)
        gen.eval()
        ok = True
    return gen, ok

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if GPT2_ADAPTER_DIR.exists():
        model = GPT2LMHeadModel.from_pretrained(str(GPT2_ADAPTER_DIR)).to(DEVICE)
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    return model, tokenizer

def gan_tensor_to_uint8_np(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu()
    while x.ndim > 2:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"Esperaba imagen 2D, obtuve {tuple(x.shape)}")
    x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = (x * 255.0).round().byte()
    return x.numpy()

def tensor_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    arr = gan_tensor_to_uint8_np(t)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")

def generate_images(gen: Generator, n: int, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(n, LATENT_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake = gen(noise)
    return fake

def generate_descriptions(model, tokenizer, prompt: str, n: int, max_len: int, temperature: float):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_len,
            num_return_sequences=n,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

CATEGORY_SEEDS = {
    "Vestido": 10,
    "Camiseta": 20,
    "Pantalón": 30,
    "Suéter": 40,
    "Abrigo": 50,
}