# 👗 Generador de Productos de Moda con IA

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


**Aplicación de IA generativa para prototipar catálogos de moda: imágenes sintéticas + descripciones de marketing**

[Descripción](#-descripción-del-proyecto) • [Características](#-características) • [Instalación](#-instalación) • [Guía de uso](#-guía-de-uso) • [Arquitectura](#️-arquitectura-técnica) • [Solución de problemas​](#-solución-de-problemas)

</div>

---

## 📋 Descripción del Proyecto

### Problemática

Equipos de diseño y marketing en moda necesitan continuamente:

- Imágenes coherentes de prendas (camisetas, vestidos, abrigos, etc.) para prototipos de catálogos.​

- Descripciones de producto consistentes para pruebas de campañas y fichas de e‑commerce.​

Producir estos materiales de forma manual consume tiempo y recursos, especialmente en fases de exploración creativa.

### Solución

Este proyecto implementa una aplicación web que:

- Genera imágenes sintéticas de productos de moda usando un DCGAN entrenado sobre Fashion‑MNIST.​

- Genera descripciones de marketing para esas categorías usando un modelo GPT‑2 adaptado al dominio de moda.​

Permite explorar rápidamente combinaciones imagen+texto para:

- Bocetos de catálogos.

- Propuestas de campañas.

- Material educativo sobre IA generativa aplicada a marketing.

### ¿Qué la hace diferente?

- 🎨 Doble canal creativo: genera tanto la parte visual (prenda) como el copy de producto.​

- 🧩 Modo híbrido: combina automáticamente una imagen sintética con una descripción alineada a la categoría (vestido, abrigo, etc.).​

- 🧪 Enfoque educativo: pensada para cursos de Deep Learning y aplicaciones de IA generativa.​

- ⚙️ Modelo desacoplado: el generador visual (DCGAN) se entrena una sola vez y luego se reutiliza en local.​

---

## ✨ Características

### Funcionales

- 🖼️ Generación de imágenes de moda a partir de ruido latente (zapatos, bolsos, prendas, etc. en estilo Fashion‑MNIST).​

- 📝 Generación de descripciones de producto basadas en GPT‑2 (ej. “Vestido formal en tono azul con detalles encaje”).​

- 🎨 Modo híbrido: crea “productos completos” (imagen + descripción + SKU y precio sugerido).​

- 🎛️ Controles en la barra lateral:

    - Número de productos a generar.

    - Categoría objetivo (vestido, camiseta, pantalón, suéter, abrigo).

    - Uso de semillas para reproducibilidad.​

### Técnicas

- 🤖 DCGAN para generación visual, entrenado sobre Fashion‑MNIST (64×64, escala de grises).​

- ✍️ GPT‑2 (Hugging Face) fine‑tuned sobre un corpus de descripciones sintéticas de moda.​

- 🌐 Interfaz web con Streamlit, organizada en tres modos: Imágenes, Descripciones, Híbrido.​

- 🧱 Arquitectura modular:

    - model.py → arquitectura del generador.

    - generator_service.py → carga de modelos y lógica de generación.

    - app_streamlit.py → UI.

---

## 🛠️ Arquitectura Técnica

### Stack Tecnológico

| Componente     | Tecnología           | Versión recomendada |
| -------------- | -------------------- | ------------------- |
| Lenguaje       | Python               | 3.10+               |
| Framework ML   | PyTorch              | 2.2+                |
| UI             | Streamlit            | 1.38+               |
| NLP            | transformers (GPT‑2) | 4.40+               |
| Dataset visual | Fashion‑MNIST        | -                   |

### Arquitectura del Modelo Visual (DCGAN)

- **Entrada**: vector de ruido latente (100 dimensiones)

- **Cuerpo**: 5 capas ConvTranspose2d con BatchNorm y ReLU.​
- **Salida**: imagen 1×64×64 en escala de grises, activación final Tanh [−1,1].​

Esquema:

```bash
Input: Z (100, 1, 1)
  ↓ ConvTranspose2d + BatchNorm + ReLU  → (512, 4, 4)
  ↓ ConvTranspose2d + BatchNorm + ReLU  → (256, 8, 8)
  ↓ ConvTranspose2d + BatchNorm + ReLU  → (128, 16, 16)
  ↓ ConvTranspose2d + BatchNorm + ReLU  → (64, 32, 32)
  ↓ ConvTranspose2d + Tanh              → (1, 64, 64)
```

## Estructura del proyecto

```bash
mnist-gan-app/
├── src/
│   ├── __init__.py
│   ├── app_streamlit.py      # Interfaz web principal
│   ├── config.py             # Configuración y rutas
│   ├── generator_service.py  # Carga de modelos y lógica de generación
│   └── model.py              # Arquitectura del generador DCGAN
├── model/
│   └── generator.pth         # Pesos del generador entrenado
├── pyproject.toml            # Dependencias y configuración del proyecto
├── .gitignore
└── README.md
``` 
---

## 📦 Instalación

### Requisitos previos

- **Python 3.10 o superior**.​ ([Descargar](https://www.python.org/downloads/))

- **Poetry** ([Guía de instalación](https://python-poetry.org/docs/#installation))

- **Archivo del modelo**: `generator.pth` (incluido en el repositorio)

Entorno virtual recomendado (venv, conda o Poetry).​

Archivo de modelo visual: model/generator.pth (pesos del DCGAN entrenado).​

### Pasos de Instalación

1. **Clonar el repositorio**

 ```bash
git clone https://github.com/JulioMunozIUD/mnist-gan-app.git
cd mnist-gan-app
```
2. **Crear y activar entorno virtual (ejemplo con venv)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. **Instalar dependencias**

Usando pip:

```bash
pip install -e .
```

o, si usas pyproject.toml con Poetry (similar a tu README original):​

```bash
poetry install
```

4. **Verificar el modelo entrenado**

Asegúrate de que el archivo de pesos está en su sitio:

```bash
ls model/generator.pth
```

## 🚀 Guía de Uso

### Iniciar la aplicación

Desde la raíz del proyecto:

```bash
# Con venv + pip
streamlit run src/app_streamlit.py

# O con Poetry
poetry run streamlit run src/app_streamlit.py
```

Abre en tu navegador:

```bash
http://localhost:8501
```

### Vista general de la interfaz
Al abrir la app verás:

1. Cabecera: título y breve descripción del sistema.​

2. Barra lateral:

- Selección de modo:

    - 🖼️ Imágenes

    - 📝 Descripciones

    - 🎨 Híbrido

- Parámetros (número de productos, semillas, categoría, etc.).

3. Area central: resultados generados (imágenes, textos o ambos).

### Modo 🖼️ Imágenes
Genera únicamente imágenes de productos de moda.

### Pasos:

1. En la barra lateral, selecciona “🖼️ Imágenes”.

2. Ajusta el número de productos a generar (1–16).

3. Opcional: marca “Usar semilla fija” e introduce un valor para reproducibilidad.

4. Pulsa “🎨 Generar Productos”.

Verás una cuadrícula de imágenes en escala de grises (64×64) que representan distintas prendas (camisetas, zapatos, bolsos, etc.).

### Uso típico:

- Prototipos de catálogos internos.

- Ilustrar clases sobre GANs y generación visual.

### Modo 📝 Descripciones
Genera únicamente texto de marketing.

### Pasos:

1. Selecciona “📝 **Descripciones**”.

2. Elige un tipo de producto base (Vestido, Camiseta, Pantalón, Suéter, Abrigo…).

3. Opcional: escribe un prompt personalizado (por ejemplo: “Vestido elegante de noche”).

4. Define:

- Número de descripciones a generar.

- Longitud máxima del texto.

- Nivel de creatividad (temperature).

5. Pulsa “📝 **Generar Descripciones**”.

Obtendrás varias propuestas de texto en tono descriptivo/comercial, listas para uso en fichas o inspiración de copy.

### Modo 🎨 Híbrido (Imagen + Texto)
Genera “productos completos” combinando imagen sintética y descripción.

### Pasos:

1. Selecciona “🎨 **Híbrido**”.

2. Escoge una **categoría de producto** (ej. Vestido).

3. Indica cuántos productos completos quieres generar.

4. Opcional: activa “Usar semilla fija” para controlar la variación visual.

5. Pulsa “🚀 **Generar Productos Completos**”.

Para cada producto verás:

- 🖼️ Imagen generada (estilo asociado a la categoría mediante semillas).

- 📝 Descripción de marketing condicionada por la categoría.

- 🔢 Un SKU sugerido.

- 💰 Precio estimado de ejemplo.

Ideal para:

- Presentar conceptos de IA generativa en marketing.

- Crear un pequeño “catálogo ficticio” para experimentos.

### 🔧 Configuración Avanzada

Puedes ajustar parámetros en src/config.py:

```python
LATENT_DIM = 100   # Dimensión del vector de ruido
IMG_CHANNELS = 1   # Canales de salida (1 = escala de grises)
IMAGE_SIZE = 64    # Tamaño de imagen (64x64)
```

Y modificar el mapeo de semillas por categoría en generator_service.py:

```python
CATEGORY_SEEDS = {
    "Vestido": 10,
    "Camiseta": 20,
    "Pantalón": 30,
    "Suéter": 40,
    "Abrigo": 50,
}
```
⚠️ Cambia estos valores solo si entiendes el impacto sobre el modelo y, en el caso visual, has reentrenado el DCGAN con la misma configuración.

## 🧪 Solución de Problemas
La aplicación no arranca (errores de importación)

### Mensaje típico:

<u>ImportError: attempted relative import with no known parent package</u>

### Causas y solución:

- Estás ejecutando app_streamlit.py desde la carpeta src.

- Ejecuta siempre desde la raíz del proyecto:

```bash
cd mnist-gan-app
streamlit run src/app_streamlit.py
```

y usa importaciones sin punto (absolutas) dentro de los módulos, como ya se configuró en este proyecto, para evitar imports relativos problemáticos.​

**Error al cargar el modelo** (<u>EOFError o FileNotFoundError</u>)

### Mensajes típicos:

- EOFError al hacer torch.load.

- FileNotFoundError: No such file or directory: 'model/generator.pth'.

### Solución:

1. Verifica que existe el archivo y tiene tamaño razonable:

```bash
ls -lh model/generator.pth
```

2. Si no existe o está corrupto:

- Vuelve a exportar generator.pth desde Colab (o desde tu entorno de entrenamiento).

- Copia el archivo a la carpeta model/.

### Las imágenes se ven “planas” o poco variadas

Recuerda que el FID aproximado del modelo es alto (~156), lo que indica que la calidad aún está lejos de ser “estado del arte”. Esto es aceptable para fines educativos, pero puedes mejorar:

- Entrenando más épocas.

- Ajustando hiperparámetros (learning rate, balance G/D).

- Usando arquitecturas más modernas (StyleGAN, difusión).

Puedes documentar estas mejoras en tu informe como trabajo futuro.

---

## 📚 Contexto Académico

Este proyecto fue desarrollado como parte de la **Evidencia de Aprendizaje 3** del curso de IA Generativa, demostrando:

- ✅ Implementación de un modelo generativo (DCGAN)
- ✅ Despliegue de una aplicación web funcional
- ✅ Solución a una problemática real (material educativo)
- ✅ Documentación técnica completa

---

## 📚 Créditos y Licencia

DCGAN entrenamiento basado en ejemplos clásicos sobre MNIST/Fashion‑MNIST.​

GPT‑2 provisto por Hugging Face Transformers.

---
