import numpy as np
import streamlit as st
from config import DEVICE
from generator_service import (
    load_generator,
    load_gpt2,
    tensor_to_pil_rgb,
    generate_images,
    generate_descriptions,
    CATEGORY_SEEDS,
)

st.set_page_config(
    page_title="Generador de Moda con IA",
    page_icon="🎨",
    layout="wide",
)

@st.cache_resource
def init_models():
    gen, gen_ok = load_generator()
    gpt2, tokenizer = load_gpt2()
    return gen, gen_ok, gpt2, tokenizer

generator, generator_ready, model_gpt2, tokenizer = init_models()

def sidebar():
    st.sidebar.header("⚙️ Configuración")
    st.sidebar.info(f"Dispositivo: **{DEVICE}**")
    st.sidebar.info(f"Parámetros Generator: **{sum(p.numel() for p in generator.parameters()):,}**")
    if generator_ready:
        st.sidebar.success("Generator cargado desde /model/generator.pth")
    else:
        st.sidebar.error("Generator NO está listo (faltan pesos entrenados).")
    mode = st.sidebar.radio(
        "Modo de uso",
        ["🖼️ Imágenes", "📝 Descripciones", "🎨 Híbrido"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Dominio: **Diseño/Marketing de moda**\n\n"
        "Visual: DCGAN sobre Fashion‑MNIST\n"
        "Texto: GPT‑2 adaptado a descripciones de productos."
    )
    return mode

def ui_images():
    st.header("🖼️ Generación de Imágenes de Productos")
    c1, c2 = st.columns([1, 3])
    with c1:
        n = st.slider("Número de productos", 1, 16, 8)
        use_seed = st.checkbox("Usar semilla fija", value=False)
        seed = st.number_input("Semilla", 0, 99999, 42)
        btn = st.button("🎨 Generar Productos", use_container_width=True)
    with c2:
        if not generator_ready:
            st.error("No hay generator entrenado disponible.")
            return
        if btn:
            s = seed if use_seed else None
            fake = generate_images(generator, n, s)
            st.markdown("### Productos generados")
            cols_per_row = min(4, n)
            for i in range(0, n, cols_per_row):
                row = st.columns(cols_per_row)
                for j, col in enumerate(row):
                    idx = i + j
                    if idx >= n:
                        break
                    with col:
                        img = tensor_to_pil_rgb(fake[idx]).resize((160, 160))
                        st.image(img, caption=f"Producto {idx+1}")

def ui_text():
    st.header("📝 Generación de Descripciones de Productos")
    c1, c2 = st.columns([1, 3])
    with c1:
        base = st.selectbox(
            "Tipo de producto base",
            ["Vestido", "Camiseta", "Pantalón", "Suéter", "Abrigo",
             "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"],
        )
        custom = st.text_input(
            "Prompt (opcional)",
            placeholder="Ej: Vestido formal en tono azul con detalles elegantes",
        )
        prompt = custom if custom.strip() else base
        n = st.slider("Número de descripciones", 1, 5, 3)
        max_len = st.slider("Longitud máxima", 30, 150, 80, 10)
        temp = st.slider("Creatividad", 0.5, 1.5, 0.8, 0.1)
        btn = st.button("📝 Generar Descripciones", use_container_width=True)
    with c2:
        if btn:
            texts = generate_descriptions(model_gpt2, tokenizer, prompt, n, max_len, temp)
            st.markdown(f"### Resultados para: *{prompt}*")
            for i, t in enumerate(texts, 1):
                st.markdown(f"#### Descripción {i}")
                st.info(t)
                st.markdown("---")

def ui_hybrid():
    st.header("🎨 Generación de Producto Completo (Imagen + Descripción)")
    c1, c2 = st.columns([1, 2])
    with c1:
        category = st.selectbox(
            "Categoría del producto",
            ["Vestido", "Camiseta", "Pantalón", "Suéter", "Abrigo"],
            key="hybrid_cat",
        )
        n = st.slider("Número de productos", 1, 6, 3, key="hybrid_n")
        use_seed = st.checkbox("Usar semilla fija", value=False, key="hybrid_seed_chk")
        seed = st.number_input("Semilla híbrido", 0, 99999, 123, key="hybrid_seed")
        btn = st.button("🚀 Generar Productos Completos", use_container_width=True, key="hybrid_btn")
    with c2:
        st.info(
            "Este modo genera para cada producto:\n\n"
            "1. Imagen sintética (DCGAN)\n"
            "2. Descripción de marketing (GPT‑2)\n\n"
            "La categoría elegida condiciona la semilla visual y el texto."
        )
    if btn:
        if not generator_ready:
            st.error("No hay generator entrenado disponible.")
            return
        base_seed = CATEGORY_SEEDS.get(category, 0)
        final_seed = base_seed + (int(seed) if use_seed else 0)
        fake = generate_images(generator, n, final_seed)
        for i in range(n):
            st.markdown(f"## 🛍️ Producto {i+1}")
            ci, cd = st.columns([1, 2])
            with ci:
                img = tensor_to_pil_rgb(fake[i]).resize((220, 220))
                st.image(img, caption=f"Imagen generada para {category}")
            with cd:
                prompt = f"{category} de moda para catálogo online"
                desc = generate_descriptions(model_gpt2, tokenizer, prompt, 1, 100, 0.8)[0]
                st.markdown(f"**Categoría:** {category}")
                st.markdown("**Descripción de marketing generada:**")
                st.info(desc)
                st.markdown(f"**SKU sugerido:** PROD-{i+1:04d}")
                st.markdown(f"**Precio estimado:** ${np.random.randint(30,150)}.99")
            st.markdown("---")
        st.success(f"{n} productos híbridos generados para “{category}”.")

# ---------- MAIN ----------
mode = sidebar()

st.title("🎨 Generador de Productos de Moda con IA")
st.markdown(
    "Sistema generativo híbrido (DCGAN + GPT‑2) para apoyar la creación de "
    "contenido visual y textual en campañas de marketing de moda."
)
st.markdown("---")

if mode == "🖼️ Imágenes":
    ui_images()
elif mode == "📝 Descripciones":
    ui_text()
else:
    ui_hybrid()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Proyecto educativo · DCGAN + GPT‑2 · Fashion‑MNIST"
    "</div>",
    unsafe_allow_html=True,
)