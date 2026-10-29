import io
import re
import html
import datetime

import streamlit as st
import subprocess
import sys
subprocess.check_call([sys.executable,
"-m", "pip", "install",
"opencv-python-headless"])
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# OCR é opcional: se Tesseract não estiver instalado, a calibração manual continua disponível.
try:
    import pytesseract
    OCR_DISPONIVEL = True
except Exception:
    pytesseract = None
    OCR_DISPONIVEL = False

st.set_page_config(page_title="LAVIM — Laboratório Virtual de Microscopia", layout="wide")


# -------------------------
# Acessibilidade
# -------------------------
def falar(texto):
    """Fala texto no navegador. Escapa caracteres para evitar quebrar o JavaScript."""
    texto_seguro = html.escape(str(texto), quote=True)
    texto_js = (
        str(texto)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", " ")
    )
    components_html = f"""
    <script>
    const texto = '{texto_js}';
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const msg = new SpeechSynthesisUtterance(texto);
        msg.lang = 'pt-BR';
        window.speechSynthesis.speak(msg);
    }}
    </script>
    """
    st.components.v1.html(components_html, height=0)


# -------------------------
# Leitura e normalização de unidades
# -------------------------
PADRAO_ESCALA = re.compile(
    r'(?<![\w])(\d+(?:[.,]\d+)?)\s*(µm|μm|um|micrometros?|mm|milimetros?|nm|nanometros?)\b',
    re.IGNORECASE
)

def para_micrometros(valor_texto):
    """Recebe, por exemplo, '500 µm', '0,5 mm' ou '200 nm'."""
    match = PADRAO_ESCALA.search(valor_texto.replace("μ", "µ"))
    if not match:
        raise ValueError("Use um valor como 500 µm, 0,5 mm ou 200 nm.")
    numero, unidade = match.groups()
    valor = float(numero.replace(",", "."))
    u = unidade.lower()
    if u.startswith("mm") or u.startswith("milimetro"):
        valor *= 1000
    elif u.startswith("nm") or u.startswith("nanometro"):
        valor /= 1000
    # µm, μm, um e micrômetro já estão em µm
    if valor <= 0:
        raise ValueError("O valor da barra deve ser maior que zero.")
    return valor, match.group(0)


# -------------------------
# Identificação auxiliar da barra
# -------------------------
def detectar_barras_candidatas(img_bgr):
    """
    Procura linhas horizontais compridas. É apenas uma sugestão:
    a pessoa deve conferir visualmente a candidata antes de usar a calibração.
    """
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = cinza.shape
    # Procura em toda a imagem, não apenas no canto inferior direito.
    # Realça estruturas horizontais escuras e claras.
    candidatos = []
    for polaridade in ("escura", "clara"):
        if polaridade == "escura":
            binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        else:
            binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel_largura = max(15, w // 18)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_largura, 1))
        linhas = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
        contornos, _ = cv2.findContours(linhas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            x, y, bw, bh = cv2.boundingRect(contorno)
            if bw >= max(20, int(w * 0.04)) and bh <= max(12, int(h * 0.025)):
                candidatos.append({
                    "x": x, "y": y, "largura_px": bw, "altura_px": bh,
                    "polaridade": polaridade
                })

    # Remove candidatos quase idênticos e prioriza linhas compridas.
    candidatos.sort(key=lambda c: c["largura_px"], reverse=True)
    unicos = []
    for c in candidatos:
        parecido = any(
            abs(c["x"] - u["x"]) < 5 and abs(c["y"] - u["y"]) < 5
            and abs(c["largura_px"] - u["largura_px"]) < 10
            for u in unicos
        )
        if not parecido:
            unicos.append(c)
    return unicos[:20]


def imagem_com_candidatos(img_bgr, candidatos):
    saida = img_bgr.copy()
    for i, c in enumerate(candidatos):
        x, y, largura, altura = c["x"], c["y"], c["largura_px"], c["altura_px"]
        cv2.rectangle(saida, (x, y), (x + largura, y + max(altura, 3)), (0, 255, 255), 2)
        cv2.putText(saida, str(i + 1), (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return saida


def ler_escala_ocr(img_bgr):
    if not OCR_DISPONIVEL:
        return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        texto = pytesseract.image_to_string(rgb, config="--psm 11")
        m = PADRAO_ESCALA.search(texto.replace("μ", "µ"))
        return m.group(0) if m else None
    except Exception:
        return None


# -------------------------
# Análise de partículas
# -------------------------
def analisar_particulas(img_bgr, um_por_pixel, area_min_px):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    # A polaridade pode variar conforme a imagem; disponibilizamos a máscara invertida
    # como padrão, apropriado para objetos escuros sobre fundo claro.
    _, binaria = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dados, centros = [], []
    for contorno in contornos:
        area_px = cv2.contourArea(contorno)
        if area_px < area_min_px:
            continue
        perimetro_px = cv2.arcLength(contorno, True)
        if perimetro_px <= 0:
            continue
        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        cx = momentos["m10"] / momentos["m00"]
        cy = momentos["m01"] / momentos["m00"]
        x, y, w, h = cv2.boundingRect(contorno)
        circularidade = (4 * np.pi * area_px) / (perimetro_px ** 2)

        dados.append({
            "ID": len(dados) + 1,
            "Área (µm²)": area_px * (um_por_pixel ** 2),
            "Comprimento do contorno (µm)": perimetro_px * um_por_pixel,
            "Largura do retângulo (µm)": w * um_por_pixel,
            "Altura do retângulo (µm)": h * um_por_pixel,
            "Circularidade": min(circularidade, 1.0),
            "Centro X (px)": cx,
            "Centro Y (px)": cy,
        })
        centros.append((cy, cx))

    df = pd.DataFrame(dados)
    dist_media = np.nan
    if len(centros) > 1:
        distancias = [
            distance.euclidean(centros[i], centros[j]) * um_por_pixel
            for i in range(len(centros))
            for j in range(i + 1, len(centros))
        ]
        if distancias:
            dist_media = float(np.mean(distancias))
    return df, dist_media, mascara


# -------------------------
# Interface
# -------------------------
st.title("LAVIM — Laboratório Virtual de Microscopia")
st.write(
    "Carregue uma imagem de microscopia. Confira a barra de escala e calibre antes "
    "de medir partículas. A detecção automática é uma sugestão e deve ser validada."
)

uploaded_file = st.file_uploader(
    "Imagem da amostra", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

if uploaded_file:
    bytes_imagem = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(bytes_imagem, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Não foi possível abrir a imagem. Tente salvar como PNG ou JPG.")
        st.stop()

    altura, largura = img.shape[:2]
    st.caption(f"Dimensões da imagem: {largura} × {altura} pixels")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagem original", use_container_width=True)

    st.header("1. Calibração da barra de escala")
    st.info(
        "Importante: o LAVIM não deve presumir que a barra tem um número fixo de pixels. "
        "A escala é calculada por: µm/px = comprimento real da barra (µm) ÷ comprimento "
        "da barra (px)."
    )

    texto_ocr = ler_escala_ocr(img)
    if texto_ocr:
        st.write(f"Leitura sugerida pelo OCR: **{texto_ocr}**")
    elif not OCR_DISPONIVEL:
        st.caption("OCR indisponível neste ambiente. Você ainda pode calibrar manualmente.")
    else:
        st.caption("O OCR não reconheceu a escala com segurança. Digite o valor impresso na barra.")

    valor_barra = st.text_input(
        "Comprimento real escrito na barra",
        value=texto_ocr or "",
        placeholder="Ex.: 500 µm ou 0,5 mm"
    )

    candidatos = detectar_barras_candidatas(img)
    usar_candidata = False
    pixels_barra = None

    if candidatos:
        st.write("Linhas horizontais candidatas encontradas (numeradas na imagem):")
        st.image(
            cv2.cvtColor(imagem_com_candidatos(img, candidatos), cv2.COLOR_BGR2RGB),
            caption="Confira qual retângulo corresponde à barra de escala.",
            use_container_width=True
        )
        opcoes = [
            f"Candidata {i+1}: {c['largura_px']} px de largura, posição x={c['x']}, y={c['y']}"
            for i, c in enumerate(candidatos)
        ]
        escolha = st.selectbox("Selecione a barra correta, se ela estiver listada", 
                               ["Não usar as candidatas"] + opcoes)
        if escolha != "Não usar as candidatas":
            indice = opcoes.index(escolha)
            pixels_barra = int(candidatos[indice]["largura_px"])
            usar_candidata = True

    st.write("Se nenhuma candidata for a barra, informe a largura medida em pixels.")
    pixels_manual = st.number_input(
        "Comprimento da barra em pixels (não invente um valor; use a medida real)",
        min_value=0, value=0, step=1,
        help="Use este campo se a detecção automática não selecionar a barra correta."
    )
    if pixels_manual > 0:
        pixels_barra = int(pixels_manual)

    if st.button("Calcular e confirmar calibração", type="primary"):
        try:
            valor_um, texto_lido = para_micrometros(valor_barra)
            if pixels_barra is None or pixels_barra <= 0:
                st.error("Não foi possível medir a barra. Selecione uma candidata correta ou informe os pixels.")
            else:
                um_por_pixel = valor_um / pixels_barra
                st.session_state["um_por_pixel"] = um_por_pixel
                st.session_state["calib_confirmada"] = True
                st.session_state["calib_msg"] = (
                    f"{texto_lido}; comprimento medido = {pixels_barra} px; "
                    f"escala = {um_por_pixel:.6g} µm/px."
                )
        except ValueError as e:
            st.session_state["calib_confirmada"] = False
            st.error(str(e))

    if st.session_state.get("calib_confirmada", False):
        st.success(st.session_state["calib_msg"])
        st.write(f"**Escala calculada:** {st.session_state['um_por_pixel']:.6g} µm/px")
        st.caption(
            "A calibração só é válida para esta imagem, resolução e configuração de aquisição. "
            "Não redimensione a imagem depois de calibrar."
        )

        st.header("2. Análise morfométrica")
        polaridade = st.radio(
            "Como os grãos aparecem em relação ao fundo?",
            ["Grãos escuros sobre fundo claro", "Grãos claros sobre fundo escuro"],
            horizontal=True
        )
        area_min_px = st.number_input(
            "Área mínima para aceitar uma partícula (px²)",
            min_value=1, value=100, step=25
        )

        if st.button("Iniciar análise"):
            # Para grãos claros em fundo escuro, inverte a imagem antes da segmentação.
            img_analise = img.copy()
            if polaridade == "Grãos claros sobre fundo escuro":
                img_analise = cv2.bitwise_not(img_analise)

            df, dist_media, mascara = analisar_particulas(
                img_analise, st.session_state["um_por_pixel"], int(area_min_px)
            )
            st.session_state["df_resultado"] = df
            st.session_state["dist_media"] = dist_media
            st.session_state["mascara"] = mascara

        if "df_resultado" in st.session_state:
            df = st.session_state["df_resultado"]
            if df.empty:
                st.warning(
                    "Nenhuma partícula foi detectada com os parâmetros atuais. "
                    "Confira a polaridade, reduza a área mínima ou teste outra imagem. "
                    "Isso não significa necessariamente que a amostra não tenha partículas."
                )
                st.image(st.session_state["mascara"], caption="Máscara binária usada na análise")
            else:
                st.success(f"{len(df)} partículas detectadas.")
                st.write(f"Distância média entre centros (pares de partículas): {st.session_state['dist_media']:.3f} µm")
                st.dataframe(df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    ax1.hist(df["Área (µm²)"], bins="auto", edgecolor="black")
                    ax1.set_xlabel("Área (µm²)")
                    ax1.set_ylabel("Frequência")
                    ax1.set_title("Distribuição da área")
                    fig1.tight_layout()
                    st.pyplot(fig1)
                    buf1 = io.BytesIO()
                    fig1.savefig(buf1, format="png", dpi=200, bbox_inches="tight")
                    st.download_button("Baixar histograma (PNG)", buf1.getvalue(),
                                       "lavim_distribuicao_area.png", "image/png")
                    plt.close(fig1)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.scatter(df["Área (µm²)"], df["Comprimento do contorno (µm)"])
                    ax2.set_xlabel("Área (µm²)")
                    ax2.set_ylabel("Comprimento do contorno (µm)")
                    ax2.set_title("Área × comprimento do contorno")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="png", dpi=200, bbox_inches="tight")
                    st.download_button("Baixar dispersão (PNG)", buf2.getvalue(),
                                       "lavim_area_vs_comprimento.png", "image/png")
                    plt.close(fig2)

                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Baixar tabela CSV", csv, "lavim_particulas.csv", "text/csv")

                st.subheader("Relatório resumido")
                st.write(
                    f"Imagem analisada em {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}. "
                    f"Foram identificadas {len(df)} partículas. "
                    f"A área média foi {df['Área (µm²)'].mean():.3f} µm² e o desvio-padrão "
                    f"foi {df['Área (µm²)'].std():.3f} µm²."
                )
else:
    st.caption("Envie uma imagem para começar.")
