from pathlib import Path
import torch

# Rutas base
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "model"

# Archivos de modelo
GENERATOR_WEIGHTS_PATH = MODEL_DIR / "generator.pth"
GPT2_ADAPTER_DIR = PROJECT_ROOT / "gpt2_fashion_final"  # opcional

# Parámetros del modelo
LATENT_DIM = 100
IMG_CHANNELS = 1
IMAGE_SIZE = 64

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")