import os
import sys
import json
import hashlib
import multiprocessing
from decimal import Decimal, getcontext
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from numba import njit

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QLabel, QStatusBar,
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QTextEdit,
    QSizePolicy, QGridLayout, QFrame, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


# ============================================================
#   GESTION DE CLAVES (Diffie-Hellman + HKDF)
# ============================================================

getcontext().prec = 50

_DOS = Decimal('2')
_UNO = Decimal('1')
_MEDIO = Decimal('0.5')
_D256 = Decimal('256')

NOMBRE_ARCHIVO_CLAVES = "claves.txt"


def _ruta_archivo_claves():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, NOMBRE_ARCHIVO_CLAVES)


def _derivar_clave(semilla, contexto):
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=contexto.encode(),
    )
    return hkdf.derive(semilla)


def _generar_claves_nuevas():
    priv_A = ec.generate_private_key(ec.SECP256K1())
    priv_B = ec.generate_private_key(ec.SECP256K1())

    pub_A = priv_A.public_key()
    pub_B = priv_B.public_key()

    secreto_A = priv_A.exchange(ec.ECDH(), pub_B)
    secreto_B = priv_B.exchange(ec.ECDH(), pub_A)

    if secreto_A != secreto_B:
        raise RuntimeError("Los secretos compartidos ECDH no coinciden.")

    semilla_maestra = hashlib.sha256(secreto_A).digest()

    clave_permutacion = _derivar_clave(semilla_maestra, "permutacion")
    clave_sustitucion = _derivar_clave(semilla_maestra, "sustitucion")
    clave_difusion = _derivar_clave(semilla_maestra, "difusion")

    return {
        "secreto_compartido": secreto_A,
        "semilla_maestra": semilla_maestra,
        "clave_permutacion": clave_permutacion,
        "clave_sustitucion": clave_sustitucion,
        "clave_difusion": clave_difusion,
    }


def _guardar_claves(claves, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Archivo de claves generado automáticamente.\n")
        f.write(f"secreto_compartido={claves['secreto_compartido'].hex()}\n")
        f.write(f"semilla_maestra={claves['semilla_maestra'].hex()}\n")
        f.write(f"clave_permutacion={claves['clave_permutacion'].hex()}\n")
        f.write(f"clave_sustitucion={claves['clave_sustitucion'].hex()}\n")
        f.write(f"clave_difusion={claves['clave_difusion'].hex()}\n")


def _leer_claves(ruta):
    datos = {}
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            datos[clave.strip()] = bytes.fromhex(valor.strip())
    requeridas = [
        "secreto_compartido", "semilla_maestra", "clave_permutacion",
        "clave_sustitucion", "clave_difusion",
    ]
    for r in requeridas:
        if r not in datos:
            raise ValueError(f"El archivo de claves está incompleto: falta '{r}'.")
    return datos


def obtener_claves():
    ruta = _ruta_archivo_claves()
    if os.path.exists(ruta):
        claves = _leer_claves(ruta)
        claves["recien_generadas"] = False
        claves["ruta_archivo"] = ruta
        return claves

    claves = _generar_claves_nuevas()
    _guardar_claves(claves, ruta)
    claves["recien_generadas"] = True
    claves["ruta_archivo"] = ruta
    return claves


# ============================================================
#    CIFRADO (Arnold + Tent + Logístico)
# ============================================================

# ── Arnold cifrado ───────────────────────────────────────────
@njit
def _arnold_cifrado_jit(canal, p, q, iteraciones, N):
    dest_i = np.zeros((N, N), dtype=np.int64)
    dest_j = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        for c in range(N):
            xi, xj = r, c
            for _ in range(iteraciones):
                xi_n = (xi + p * xj) % N
                xj_n = (q * xi + (p * q + 1) * xj) % N
                xi, xj = xi_n, xj_n
            dest_i[r, c] = xi
            dest_j[r, c] = xj
    resultado = np.zeros_like(canal)
    for r in range(N):
        for c in range(N):
            resultado[dest_i[r, c], dest_j[r, c]] = canal[r, c]
    return resultado


# ── Arnold descifrado ────────────────────────────────────────
@njit
def _arnold_descifrado_jit(canal, p, q, iteraciones, N):
    pq1 = p * q + 1
    orig_i = np.zeros((N, N), dtype=np.int64)
    orig_j = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        for c in range(N):
            xi, xj = r, c
            for _ in range(iteraciones):
                xi_n = (pq1 * xi - p * xj) % N
                xj_n = (-q * xi + xj) % N
                xi, xj = xi_n, xj_n
            orig_i[r, c] = xi
            orig_j[r, c] = xj
    resultado = np.zeros_like(canal)
    for r in range(N):
        for c in range(N):
            resultado[orig_i[r, c], orig_j[r, c]] = canal[r, c]
    return resultado


# ── Logístico (Difusión Bidireccional) ──────
@njit
def _generar_seq_logistico_jit(x0, L):
    seq = np.empty(L, dtype=np.uint8)
    x = x0
    for i in range(L):
        x = 4.0 * x * (1.0 - x)
        seq[i] = int(x * 256) % 256
    return seq


@njit
def _dif_forward_add_jit(datos, seq):
    """Pasada hacia adelante: c[i] = (p[i] + seq[i] + c[i-1]) mod 256"""
    L = len(datos)
    res = np.empty(L, dtype=np.uint8)
    res[0] = (int(datos[0]) + int(seq[0])) % 256
    for i in range(1, L):
        res[i] = (int(datos[i]) + int(seq[i]) + int(res[i - 1])) % 256
    return res


@njit
def _dif_forward_add_inv_jit(inter, seq):
    """Inversa de la pasada hacia adelante"""
    L = len(inter)
    res = np.empty(L, dtype=np.uint8)
    res[0] = (int(inter[0]) - int(seq[0])) % 256
    for i in range(1, L):
        res[i] = (int(inter[i]) - int(seq[i]) - int(inter[i - 1])) % 256
    return res


@njit
def _dif_backward_add_jit(datos, seq):
    """Pasada hacia atras: c[i] = (p[i] + seq[i] + c[i+1]) mod 256"""
    L = len(datos)
    res = np.empty(L, dtype=np.uint8)
    res[L - 1] = (int(datos[L - 1]) + int(seq[L - 1])) % 256
    for i in range(L - 2, -1, -1):
        res[i] = (int(datos[i]) + int(seq[i]) + int(res[i + 1])) % 256
    return res


@njit
def _dif_backward_add_inv_jit(cifrado, seq):
    """Inversa de la pasada hacia atras"""
    L = len(cifrado)
    res = np.empty(L, dtype=np.uint8)
    res[L - 1] = (int(cifrado[L - 1]) - int(seq[L - 1])) % 256
    for i in range(L - 2, -1, -1):
        res[i] = (int(cifrado[i]) - int(seq[i]) - int(cifrado[i + 1])) % 256
    return res


def extraer_x0_logistico(clave_bytes):
    valor = int.from_bytes(clave_bytes, 'big')
    x0 = valor / (2 ** 256)
    if x0 <= 0: x0 = 1e-15
    if x0 >= 1: x0 = 1 - 1e-15
    return np.float64(x0)


def _derivar_claves_difusion(clave, n_rondas=2):
    """Deriva sub-claves (x0 forward y x0 backward) por cada ronda para evitar reutilización."""
    x0_pares = []
    for r in range(n_rondas):
        clave_fwd = hashlib.sha256(clave + f'fwd{r}'.encode()).digest()
        clave_bwd = hashlib.sha256(clave + f'bwd{r}'.encode()).digest()
        x0_pares.append((
            extraer_x0_logistico(clave_fwd),
            extraer_x0_logistico(clave_bwd),
        ))
    return x0_pares


def precompilar_jit():
    """Fuerza la compilación JIT de todas las funciones Numba."""
    c = np.zeros((4, 4), dtype=np.uint8)
    d = np.zeros(48, dtype=np.uint8)
    s = np.zeros(48, dtype=np.uint8)
    _arnold_cifrado_jit(c, np.int64(1), np.int64(1), np.int64(1), np.int64(4))
    _arnold_descifrado_jit(c, np.int64(1), np.int64(1), np.int64(1), np.int64(4))
    _generar_seq_logistico_jit(np.float64(0.5), 48)
    _dif_forward_add_jit(d, s)
    _dif_forward_add_inv_jit(d, s)
    _dif_backward_add_jit(d, s)
    _dif_backward_add_inv_jit(d, s)


# ----------------  padding ----------------

def agregar_padding_cuadrado(imagen):
    ancho, alto = imagen.size
    lado = max(ancho, alto)
    imagen_cuadrada = Image.new('RGB', (lado, lado), color='black')
    izquierda = (lado - ancho) // 2
    arriba = (lado - alto) // 2
    imagen_cuadrada.paste(imagen, (izquierda, arriba))
    return imagen_cuadrada, (ancho, alto, izquierda, arriba)


def eliminar_padding(imagen_cuadrada, info_padding):
    ancho_orig, alto_orig, izquierda, arriba = info_padding
    region = (izquierda, arriba, izquierda + ancho_orig, arriba + alto_orig)
    return imagen_cuadrada.crop(region)


# ---------------- ETAPA 1: ARNOLD (permutacion) ----------------

def extraer_parametros_arnold(clave_bytes, N):
    valor = int.from_bytes(clave_bytes, "big")
    p = np.int64(((valor >> 224) & 0xFFFFFFFF) % (N - 1) + 1)
    q = np.int64(((valor >> 192) & 0xFFFFFFFF) % (N - 1) + 1)
    it = np.int64(((valor >> 184) & 0xFF) % 10 + 1)
    return p, q, it


def cifrado_arnold(imagen, clave):
    N = imagen.size[0]
    p, q, it = extraer_parametros_arnold(clave, N)
    canales = [np.array(ch, dtype=np.uint8) for ch in imagen.split()]
    cifrados = [_arnold_cifrado_jit(ch, p, q, it, np.int64(N)) for ch in canales]
    return Image.merge(imagen.mode, [Image.fromarray(ch, "L") for ch in cifrados]), (p, q, it)


def descifrado_arnold(imagen_cifrada, clave):
    N = imagen_cifrada.size[0]
    p, q, it = extraer_parametros_arnold(clave, N)
    canales = [np.array(ch, dtype=np.uint8) for ch in imagen_cifrada.split()]
    desc = [_arnold_descifrado_jit(ch, p, q, it, np.int64(N)) for ch in canales]
    return Image.merge(imagen_cifrada.mode, [Image.fromarray(ch, "L") for ch in desc]), (p, q, it)


# ---------------- ETAPA 2: TENT (sustitucion) ----------------

def extraer_x0_tent(clave_bytes):
    return Decimal(int.from_bytes(clave_bytes, 'big')) / Decimal(2 ** 256)


def tent_paso(x):
    return _DOS * x if x < _MEDIO else _DOS * (_UNO - x)


def generar_secuencia_tent(x0, longitud):
    x = x0
    seq = np.empty(longitud, dtype=np.uint8)
    for i in range(longitud):
        x = tent_paso(x)
        seq[i] = int(x * _D256) % 256
    return seq


def _procesar_canal_tent_puro(args):
    canal_flat, x0, longitud, H, W = args
    seg_seq = generar_secuencia_tent(x0, longitud)
    resultado = (canal_flat ^ seg_seq).astype(np.uint8)
    return resultado.reshape(H, W)


def cifrado_tent(imagen, clave, usar_multiproceso=True):
    arr = np.array(imagen, dtype=np.uint8)
    H, W = arr.shape[:2]
    tam_canal = H * W

    x0_base = extraer_x0_tent(clave)
    x0_r = x0_base
    x0_g = (x0_base + Decimal('1e-15')) % Decimal('1')
    x0_b = (x0_base + Decimal('2e-15')) % Decimal('1')

    canales_flat = [arr[:, :, i].flatten() for i in range(3)]
    x0_canales = [x0_r, x0_g, x0_b]

    tareas = [(canales_flat[i], x0_canales[i], tam_canal, H, W) for i in range(3)]

    if usar_multiproceso:
        with ProcessPoolExecutor(max_workers=3) as executor:
            resultados = list(executor.map(_procesar_canal_tent_puro, tareas))
    else:
        resultados = [_procesar_canal_tent_puro(t) for t in tareas]

    arr_resultado = np.stack(resultados, axis=2)
    return Image.fromarray(arr_resultado, mode='RGB'), float(x0_base)


def descifrado_tent(imagen_cifrada, clave, usar_multiproceso=True):
    imagen, x0 = cifrado_tent(imagen_cifrada, clave, usar_multiproceso=usar_multiproceso)
    return imagen, x0


# ---------------- ETAPA 3: LOGISTICO (difusion) ----------------

def cifrado_logistico(imagen, clave, n_rondas=2):
    datos = np.array(imagen, dtype=np.uint8).flatten()
    L = len(datos)
    x0_pares = _derivar_claves_difusion(clave, n_rondas)

    resultado = datos
    for r, (x0_fwd, x0_bwd) in enumerate(x0_pares):
        seq_fwd = _generar_seq_logistico_jit(x0_fwd, L)
        seq_bwd = _generar_seq_logistico_jit(x0_bwd, L)
        resultado = _dif_forward_add_jit(resultado, seq_fwd)
        resultado = _dif_backward_add_jit(resultado, seq_bwd)

    forma = np.array(imagen).shape
    x0_log_str = f"fwd0={x0_pares[0][0]:.6f}, bwd0={x0_pares[0][1]:.6f}"
    return Image.fromarray(resultado.reshape(forma), mode=imagen.mode), x0_log_str


def descifrado_logistico(imagen_cifrada, clave, n_rondas=2):
    datos = np.array(imagen_cifrada, dtype=np.uint8).flatten()
    L = len(datos)
    x0_pares = _derivar_claves_difusion(clave, n_rondas)

    resultado = datos
    # Orden inverso: primero se deshace la última ronda backward, luego forward
    for x0_fwd, x0_bwd in reversed(x0_pares):
        seq_fwd = _generar_seq_logistico_jit(x0_fwd, L)
        seq_bwd = _generar_seq_logistico_jit(x0_bwd, L)
        resultado = _dif_backward_add_inv_jit(resultado, seq_bwd)
        resultado = _dif_forward_add_inv_jit(resultado, seq_fwd)

    forma = np.array(imagen_cifrada).shape
    x0_log_str = f"fwd0={x0_pares[0][0]:.6f}, bwd0={x0_pares[0][1]:.6f}"
    return Image.fromarray(resultado.reshape(forma), mode=imagen_cifrada.mode), x0_log_str


# ---------------- PIPELINE COMPLETO ----------------

def cifrar_imagen_completa(imagen_original, c_perm, c_sust, c_dif, usar_multiproceso=True, log=None):
    def _log(msg):
        if log: log(msg)

    if imagen_original.mode != "RGB":
        imagen_original = imagen_original.convert("RGB")

    imagen_pad, info_padding = agregar_padding_cuadrado(imagen_original)
    _log(f"Padding aplicado: {imagen_original.size} -> {imagen_pad.size}")

    imagen_arnold, (p, q, it) = cifrado_arnold(imagen_pad, c_perm)
    _log(f"Arnold aplicado (p={p}, q={q}, iter={it})")

    imagen_tent, x0_tent = cifrado_tent(imagen_arnold, c_sust, usar_multiproceso=usar_multiproceso)
    _log(f"Tent aplicado (x0={x0_tent:.10f})")

    imagen_cifrada, x0_log = cifrado_logistico(imagen_tent, c_dif)
    _log(f"Logístico aplicado (Rondas=2, {x0_log})")

    detalles = {
        "arnold": {"p": int(p), "q": int(q), "iteraciones": int(it)},
        "tent_x0": x0_tent,
        "logistico_x0": x0_log,
    }
    return imagen_cifrada, imagen_pad, info_padding, detalles


def descifrar_imagen_completa(imagen_cifrada, info_padding, c_perm, c_sust, c_dif,
                               usar_multiproceso=True, log=None):
    def _log(msg):
        if log: log(msg)

    imagen_desc_log, x0_log = descifrado_logistico(imagen_cifrada, c_dif)
    _log(f"Logístico inverso aplicado ({x0_log})")

    imagen_desc_tent, x0_tent = descifrado_tent(imagen_desc_log, c_sust, usar_multiproceso=usar_multiproceso)
    _log(f"Tent inverso aplicado (x0={x0_tent:.10f})")

    imagen_desc_arnold, (p, q, it) = descifrado_arnold(imagen_desc_tent, c_perm)
    _log(f"Arnold inverso aplicado (p={p}, q={q}, iter={it})")

    if info_padding is not None:
        imagen_final = eliminar_padding(imagen_desc_arnold, info_padding)
        _log(f"Padding recortado: {imagen_desc_arnold.size} -> {imagen_final.size}")
    else:
        imagen_final = imagen_desc_arnold
        _log("No se encontró información de padding; se muestra la imagen cuadrada completa.")

    detalles = {
        "arnold": {"p": int(p), "q": int(q), "iteraciones": int(it)},
        "tent_x0": x0_tent,
        "logistico_x0": x0_log,
    }
    return imagen_final, imagen_desc_arnold, detalles


# ============================================================
#   METRICAS DE CONSTRUCCION (MSE, PSNR, SSIM)
# ============================================================

def evaluar_descifrado(imagen_original, imagen_descifrada):
    arr_orig = np.array(imagen_original.convert("RGB"))
    arr_desc = np.array(imagen_descifrada.convert("RGB"))

    if arr_orig.shape != arr_desc.shape:
        return {"error": f"Las dimensiones no coinciden: original {arr_orig.shape[:2]} vs descifrada {arr_desc.shape[:2]}."}

    orig_f = arr_orig.astype(np.float64)
    desc_f = arr_desc.astype(np.float64)
    mse = float(np.mean((orig_f - desc_f) ** 2))

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = float(10 * np.log10((255 ** 2) / mse))

    if len(arr_orig.shape) == 3:
        ssim_val = float(ssim(arr_orig, arr_desc, channel_axis=-1, data_range=255))
    else:
        ssim_val = float(ssim(arr_orig, arr_desc, data_range=255))

    son_identicas = (mse == 0 and ssim_val == 1.0)

    if son_identicas:
        veredicto = "Descifrado PERFECTO: las imágenes son idénticas."
    elif mse < 1.0 and ssim_val > 0.9990:
        veredicto = "Descifrado con errores marginales (redondeo de flotantes)."
    else:
        veredicto = "Descifrado INCORRECTO: estructuras distorsionadas."

    return {"mse": mse, "psnr": psnr, "ssim": ssim_val, "son_identicas": son_identicas, "veredicto": veredicto}


def calcular_mapa_diferencias(imagen_original, imagen_descifrada):
    orig = np.array(imagen_original.convert("RGB")).astype(np.int16)
    desc = np.array(imagen_descifrada.convert("RGB")).astype(np.int16)
    diff = np.abs(orig - desc)
    maximo = diff.max()
    if maximo > 0:
        diff_vis = (diff / maximo * 255).astype(np.uint8)
    else:
        diff_vis = diff.astype(np.uint8)
    return diff_vis


# ============================================================
#   METRICAS DE ENTROPIA (Shannon + histogramas)
# ============================================================

NOMBRES_CANALES = ['Rojo (R)', 'Verde (G)', 'Azul (B)']
COLORES_CANALES = ['#d32f2f', '#2e7d32', '#1565c0']


def calcular_entropia_shannon(canal_plano):
    total_pixeles = len(canal_plano)
    frecuencias, _ = np.histogram(canal_plano, bins=256, range=(0, 256))
    probabilidades = frecuencias / total_pixeles
    probabilidades_positivas = probabilidades[probabilidades > 0]
    entropia = -np.sum(probabilidades_positivas * np.log2(probabilidades_positivas))
    return float(entropia)


def analizar_histogramas_entropia(imagen_original, imagen_cifrada):
    arr_orig = np.array(imagen_original.convert("RGB"))
    arr_cifr = np.array(imagen_cifrada.convert("RGB"))

    hist_original, hist_cifrada = [], []
    entropia_original, entropia_cifrada = [], []

    for i in range(3):
        canal_o = arr_orig[:, :, i].flatten()
        canal_c = arr_cifr[:, :, i].flatten()

        h_o, _ = np.histogram(canal_o, bins=256, range=(0, 256))
        h_c, _ = np.histogram(canal_c, bins=256, range=(0, 256))

        hist_original.append(h_o)
        hist_cifrada.append(h_c)
        entropia_original.append(calcular_entropia_shannon(canal_o))
        entropia_cifrada.append(calcular_entropia_shannon(canal_c))

    return {
        "canales": NOMBRES_CANALES, "colores": COLORES_CANALES,
        "hist_original": hist_original, "hist_cifrada": hist_cifrada,
        "entropia_original": entropia_original, "entropia_cifrada": entropia_cifrada,
    }


# ============================================================
#   METRICAS DE CORRELACION DE PIXELES ADYACENTES
# ============================================================

DIRECCIONES = ['Horizontal', 'Vertical', 'Diagonal']
CANALES_LETRAS = ['R', 'G', 'B']
COLORES_RGB = ['#d32f2f', '#2e7d32', '#1565c0']


def calcular_y_muestrear_correlacion(arr_canal, direccion, num_pares=2500, rng=None):
    if direccion == 'Horizontal':
        x, y = arr_canal[:, :-1].flatten(), arr_canal[:, 1:].flatten()
    elif direccion == 'Vertical':
        x, y = arr_canal[:-1, :].flatten(), arr_canal[1:, :].flatten()
    elif direccion == 'Diagonal':
        x, y = arr_canal[:-1, :-1].flatten(), arr_canal[1:, 1:].flatten()
    else:
        raise ValueError("DirecciOn no vAlida.")

    coeficiente = float(np.corrcoef(x, y)[0, 1])
    if rng is None: rng = np.random.default_rng()
    n = min(num_pares, len(x))
    idx = rng.choice(len(x), n, replace=False)
    return coeficiente, x[idx], y[idx]


def analizar_correlacion(imagen_original, imagen_cifrada, num_pares=2500, semilla_rng=42):
    arr_orig = np.array(imagen_original.convert("RGB"))
    arr_cifr = np.array(imagen_cifrada.convert("RGB"))
    rng = np.random.default_rng(semilla_rng)

    datos, tabla = {}, {}
    for arr_img, etiqueta in [(arr_orig, 'Original'), (arr_cifr, 'Encrypted')]:
        datos[etiqueta], tabla[etiqueta] = {}, {}
        for dir_name in DIRECCIONES:
            datos[etiqueta][dir_name], tabla[etiqueta][dir_name] = {}, {}
            for c_idx, letra in enumerate(CANALES_LETRAS):
                coef, x_p, y_p = calcular_y_muestrear_correlacion(arr_img[:, :, c_idx], dir_name, num_pares, rng=rng)
                datos[etiqueta][dir_name][letra] = (coef, x_p, y_p)
                tabla[etiqueta][dir_name][letra] = coef

    return {"direcciones": DIRECCIONES, "canales": CANALES_LETRAS, "colores": COLORES_RGB, "datos": datos, "tabla_coeficientes": tabla}



# ============================================================

def pil_a_pixmap(imagen_pil: Image.Image) -> QPixmap:
    if imagen_pil.mode != "RGB": imagen_pil = imagen_pil.convert("RGB")
    ancho, alto = imagen_pil.size
    datos = imagen_pil.tobytes("raw", "RGB")
    qimagen = QImage(datos, ancho, alto, ancho * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimagen.copy())


def escalar_pixmap_para_label(pixmap: QPixmap, label) -> QPixmap:
    return pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ============================================================
#   ESTADO COMPARTIDO ENTRE PESTAÑAS
# ============================================================

class EstadoApp:
    def __init__(self):
        self.claves = None
        self.cifrar_imagen_original = None
        self.cifrar_imagen_padding = None
        self.cifrar_imagen_cifrada = None
        self.cifrar_info_padding = None
        self.cifrar_ruta_original = None
        self.descifrar_imagen_cifrada = None
        self.descifrar_imagen_con_padding = None
        self.descifrar_imagen_final = None
        self.descifrar_ruta_cifrada = None

estado_global = EstadoApp()




class WorkerGenerico(QThread):
    progreso = pyqtSignal(str)
    terminado = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, funcion, *args, **kwargs):
        super().__init__()
        self._funcion = funcion
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            def _log(msg): self.progreso.emit(msg)
            if "log" in self._funcion.__code__.co_varnames:
                self._kwargs.setdefault("log", _log)
            resultado = self._funcion(*self._args, **self._kwargs)
            self.terminado.emit(resultado)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(str(exc))


# ============================================================
#    PESTAÑA "CIFRAR"
# ============================================================

class TabCifrar(QWidget):
    def __init__(self, obtener_claves_callback):
        super().__init__()
        self._obtener_claves = obtener_claves_callback
        self._worker = None
        self._ruta_imagen_actual = None
        self._construir_ui()

    def _construir_ui(self):
        layout_principal = QVBoxLayout(self)
        titulo = QLabel("Cifrado de Imagen")
        titulo.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 8px;")
        layout_principal.addWidget(titulo)

        barra_acciones = QHBoxLayout()
        self.btn_subir = QPushButton("📁 Subir Imagen")
        self.btn_subir.clicked.connect(self._subir_imagen)
        barra_acciones.addWidget(self.btn_subir)

        self.btn_cifrar = QPushButton("🔒 Cifrar Imagen")
        self.btn_cifrar.clicked.connect(self._cifrar_imagen)
        self.btn_cifrar.setEnabled(False)
        barra_acciones.addWidget(self.btn_cifrar)

        self.btn_guardar = QPushButton("💾 Guardar Imagen Cifrada")
        self.btn_guardar.clicked.connect(self._guardar_cifrada)
        self.btn_guardar.setEnabled(False)
        barra_acciones.addWidget(self.btn_guardar)
        barra_acciones.addStretch()
        layout_principal.addLayout(barra_acciones)

        area_imagenes = QHBoxLayout()
        grupo_original = QGroupBox("Imagen Original")
        v_original = QVBoxLayout(grupo_original)
        self.label_original = QLabel("Ninguna imagen cargada")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setMinimumSize(380, 380)
        self.label_original.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_original.addWidget(self.label_original)
        area_imagenes.addWidget(grupo_original)

        grupo_cifrada = QGroupBox("Imagen Cifrada")
        v_cifrada = QVBoxLayout(grupo_cifrada)
        self.label_cifrada = QLabel("Aún no se ha cifrado ninguna imagen")
        self.label_cifrada.setAlignment(Qt.AlignCenter)
        self.label_cifrada.setMinimumSize(380, 380)
        self.label_cifrada.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_cifrada.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_cifrada.addWidget(self.label_cifrada)
        area_imagenes.addWidget(grupo_cifrada)
        layout_principal.addLayout(area_imagenes, stretch=1)

        grupo_log = QGroupBox("Registro del proceso")
        v_log = QVBoxLayout(grupo_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(120)
        v_log.addWidget(self.txt_log)
        layout_principal.addWidget(grupo_log)
        self._pixmap_original = None
        self._pixmap_cifrada = None

    def _log(self, mensaje): self.txt_log.append(mensaje)

    def _subir_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: return
        try:
            imagen = Image.open(ruta).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo abrir la imagen:\n{exc}")
            return

        self._ruta_imagen_actual = ruta
        estado_global.cifrar_imagen_original = imagen
        estado_global.cifrar_ruta_original = ruta
        estado_global.cifrar_imagen_cifrada = None
        estado_global.cifrar_info_padding = None

        self._pixmap_original = pil_a_pixmap(imagen)
        self.label_original.setPixmap(escalar_pixmap_para_label(self._pixmap_original, self.label_original))
        self.label_cifrada.clear()
        self.label_cifrada.setText("Aun no se ha cifrado ninguna imagen")
        self._pixmap_cifrada = None
        self.btn_cifrar.setEnabled(True)
        self.btn_guardar.setEnabled(False)
        self.txt_log.clear()
        self._log(f"Imagen cargada: {os.path.basename(ruta)} ({imagen.size[0]}x{imagen.size[1]})")

    def _cifrar_imagen(self):
        imagen = estado_global.cifrar_imagen_original
        if imagen is None: return
        claves = self._obtener_claves()
        if claves is None:
            QMessageBox.warning(self, "Claves no disponibles", "Las claves criptográficas aun no se han cargado.")
            return

        self.btn_cifrar.setEnabled(False)
        self.btn_subir.setEnabled(False)
        self._log("Iniciando cifrado (Arnold -> Tent -> Logístico Bidireccional)...")

        self._worker = WorkerGenerico(cifrar_imagen_completa, imagen, claves["clave_permutacion"], claves["clave_sustitucion"], claves["clave_difusion"])
        self._worker.progreso.connect(self._log)
        self._worker.terminado.connect(self._al_terminar_cifrado)
        self._worker.error.connect(self._al_fallar)
        self._worker.start()

    def _al_terminar_cifrado(self, resultado):
        imagen_cifrada, imagen_pad, info_padding, detalles = resultado
        estado_global.cifrar_imagen_padding = imagen_pad
        estado_global.cifrar_imagen_cifrada = imagen_cifrada
        estado_global.cifrar_info_padding = info_padding

        self._pixmap_cifrada = pil_a_pixmap(imagen_cifrada)
        self.label_cifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_cifrada, self.label_cifrada))
        self._log("Cifrado completado.")
        self._log(f"Parámetros Arnold: {detalles['arnold']}")
        self._log(f"x0 Tent: {detalles['tent_x0']:.10f}")
        self._log(f"Logístico: {detalles['logistico_x0']}")
        self.btn_cifrar.setEnabled(True)
        self.btn_subir.setEnabled(True)
        self.btn_guardar.setEnabled(True)

    def _al_fallar(self, mensaje_error):
        QMessageBox.critical(self, "Error durante el cifrado", mensaje_error)
        self.btn_cifrar.setEnabled(True)
        self.btn_subir.setEnabled(True)

    def _guardar_cifrada(self):
        imagen_cifrada = estado_global.cifrar_imagen_cifrada
        if imagen_cifrada is None: return
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar imagen cifrada", "imagen_cifrada.png", "Imagen PNG (*.png)")
        if not ruta: return
        if not ruta.lower().endswith(".png"): ruta += ".png"
        try:
            imagen_cifrada.save(ruta)
            info_padding = estado_global.cifrar_info_padding
            if info_padding is not None:
                ruta_meta = ruta + ".padding.json"
                ancho, alto, izquierda, arriba = info_padding
                with open(ruta_meta, "w", encoding="utf-8") as f:
                    json.dump({"ancho": ancho, "alto": alto, "izquierda": izquierda, "arriba": arriba}, f)
                self._log(f"Metadatos de padding guardados en: {os.path.basename(ruta_meta)}")
            self._log(f"Imagen cifrada guardada en: {ruta}")
            QMessageBox.information(self, "Guardado exitoso", f"Imagen cifrada guardada en:\n{ruta}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al guardar", str(exc))

    def actualizar_al_mostrar(self):
        if self._pixmap_original is not None:
            self.label_original.setPixmap(escalar_pixmap_para_label(self._pixmap_original, self.label_original))
        if self._pixmap_cifrada is not None:
            self.label_cifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_cifrada, self.label_cifrada))


# ============================================================
#   PESTAÑA "DESCIFRAR"
# ============================================================

class TabDescifrar(QWidget):
    def __init__(self, obtener_claves_callback):
        super().__init__()
        self._obtener_claves = obtener_claves_callback
        self._worker = None
        self._info_padding_cargada = None
        self._construir_ui()

    def _construir_ui(self):
        layout_principal = QVBoxLayout(self)
        titulo = QLabel("Descifrado de Imagen")
        titulo.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 8px;")
        layout_principal.addWidget(titulo)

        barra_acciones = QHBoxLayout()
        self.btn_subir = QPushButton("📁 Subir Imagen Cifrada")
        self.btn_subir.clicked.connect(self._subir_imagen)
        barra_acciones.addWidget(self.btn_subir)

        self.btn_descifrar = QPushButton("🔓 Descifrar Imagen")
        self.btn_descifrar.clicked.connect(self._descifrar_imagen)
        self.btn_descifrar.setEnabled(False)
        barra_acciones.addWidget(self.btn_descifrar)

        self.btn_guardar = QPushButton("💾 Guardar Imagen Descifrada")
        self.btn_guardar.clicked.connect(self._guardar_descifrada)
        self.btn_guardar.setEnabled(False)
        barra_acciones.addWidget(self.btn_guardar)
        barra_acciones.addStretch()
        layout_principal.addLayout(barra_acciones)

        area_imagenes = QHBoxLayout()
        grupo_cifrada = QGroupBox("Imagen Cifrada")
        v_cifrada = QVBoxLayout(grupo_cifrada)
        self.label_cifrada = QLabel("Ninguna imagen cargada")
        self.label_cifrada.setAlignment(Qt.AlignCenter)
        self.label_cifrada.setMinimumSize(380, 380)
        self.label_cifrada.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_cifrada.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_cifrada.addWidget(self.label_cifrada)
        area_imagenes.addWidget(grupo_cifrada)

        grupo_descifrada = QGroupBox("Imagen Descifrada")
        v_descifrada = QVBoxLayout(grupo_descifrada)
        self.label_descifrada = QLabel("Aún no se ha descifrado ninguna imagen")
        self.label_descifrada.setAlignment(Qt.AlignCenter)
        self.label_descifrada.setMinimumSize(380, 380)
        self.label_descifrada.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_descifrada.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_descifrada.addWidget(self.label_descifrada)
        area_imagenes.addWidget(grupo_descifrada)
        layout_principal.addLayout(area_imagenes, stretch=1)

        grupo_log = QGroupBox("Registro del proceso")
        v_log = QVBoxLayout(grupo_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(120)
        v_log.addWidget(self.txt_log)
        layout_principal.addWidget(grupo_log)
        self._pixmap_cifrada = None
        self._pixmap_descifrada = None

    def _log(self, mensaje): self.txt_log.append(mensaje)

    def _subir_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen cifrada", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: return
        try:
            imagen = Image.open(ruta).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo abrir la imagen:\n{exc}")
            return

        estado_global.descifrar_imagen_cifrada = imagen
        estado_global.descifrar_ruta_cifrada = ruta
        estado_global.descifrar_imagen_final = None
        self.txt_log.clear()
        self._log(f"Imagen cargada: {os.path.basename(ruta)} ({imagen.size[0]}x{imagen.size[1]})")

        ruta_meta = ruta + ".padding.json"
        self._info_padding_cargada = None
        if os.path.exists(ruta_meta):
            try:
                with open(ruta_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._info_padding_cargada = (meta["ancho"], meta["alto"], meta["izquierda"], meta["arriba"])
                self._log("Metadatos de padding encontrados: se recortará automáticamente.")
            except Exception:
                self._log("No se pudieron leer los metadatos de padding; se continuará sin recortar.")
        else:
            if estado_global.cifrar_imagen_cifrada is not None and estado_global.cifrar_info_padding is not None:
                self._info_padding_cargada = estado_global.cifrar_info_padding
                self._log("Usando información de padding de la última imagen cifrada en esta sesión.")
            else:
                self._log("No se encontró información de padding; el resultado se mostrará en formato cuadrado.")

        self._pixmap_cifrada = pil_a_pixmap(imagen)
        self.label_cifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_cifrada, self.label_cifrada))
        self.label_descifrada.clear()
        self.label_descifrada.setText("Aún no se ha descifrado ninguna imagen")
        self._pixmap_descifrada = None
        self.btn_descifrar.setEnabled(True)
        self.btn_guardar.setEnabled(False)

    def _descifrar_imagen(self):
        imagen = estado_global.descifrar_imagen_cifrada
        if imagen is None: return
        claves = self._obtener_claves()
        if claves is None:
            QMessageBox.warning(self, "Claves no disponibles", "Las claves criptográficas aún no se han cargado.")
            return

        self.btn_descifrar.setEnabled(False)
        self.btn_subir.setEnabled(False)
        self._log("Iniciando descifrado (Logístico⁻¹ -> Tent⁻¹ -> Arnold⁻¹)...")

        self._worker = WorkerGenerico(descifrar_imagen_completa, imagen, self._info_padding_cargada, claves["clave_permutacion"], claves["clave_sustitucion"], claves["clave_difusion"])
        self._worker.progreso.connect(self._log)
        self._worker.terminado.connect(self._al_terminar_descifrado)
        self._worker.error.connect(self._al_fallar)
        self._worker.start()

    def _al_terminar_descifrado(self, resultado):
        imagen_final, imagen_desc_con_padding, detalles = resultado
        estado_global.descifrar_imagen_con_padding = imagen_desc_con_padding
        estado_global.descifrar_imagen_final = imagen_final

        self._pixmap_descifrada = pil_a_pixmap(imagen_final)
        self.label_descifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_descifrada, self.label_descifrada))
        self._log("Descifrado completado.")
        self._log(f"Parámetros Arnold: {detalles['arnold']}")
        self.btn_descifrar.setEnabled(True)
        self.btn_subir.setEnabled(True)
        self.btn_guardar.setEnabled(True)

    def _al_fallar(self, mensaje_error):
        QMessageBox.critical(self, "Error durante el descifrado", mensaje_error)
        self.btn_descifrar.setEnabled(True)
        self.btn_subir.setEnabled(True)

    def _guardar_descifrada(self):
        imagen_final = estado_global.descifrar_imagen_final
        if imagen_final is None: return
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar imagen descifrada", "imagen_descifrada.png", "Imagen PNG (*.png)")
        if not ruta: return
        if not ruta.lower().endswith(".png"): ruta += ".png"
        try:
            imagen_final.save(ruta)
            QMessageBox.information(self, "Guardado exitoso", f"Imagen descifrada guardada en:\n{ruta}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al guardar", str(exc))

    def actualizar_al_mostrar(self):
        if self._pixmap_cifrada is not None:
            self.label_cifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_cifrada, self.label_cifrada))
        if self._pixmap_descifrada is not None:
            self.label_descifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_descifrada, self.label_descifrada))


# ============================================================
#   PESTAÑA "METRICA DE CONSTRUCCION"
# ============================================================

class TarjetaMetrica(QFrame):
    def __init__(self, titulo):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #ffffff; border: 1px solid #d0d0d0; border-radius: 6px; padding: 6px; }")
        layout = QVBoxLayout(self)
        self.lbl_titulo = QLabel(titulo)
        self.lbl_titulo.setAlignment(Qt.AlignCenter)
        self.lbl_titulo.setStyleSheet("font-size: 12px; color: #555;")
        self.lbl_valor = QLabel("—")
        self.lbl_valor.setAlignment(Qt.AlignCenter)
        self.lbl_valor.setFont(QFont("", 20, QFont.Bold))
        layout.addWidget(self.lbl_titulo)
        layout.addWidget(self.lbl_valor)

    def set_valor(self, texto): self.lbl_valor.setText(texto)


class TabMetricaConstruccion(QWidget):
    def __init__(self):
        super().__init__()
        self._imagen_original = None
        self._imagen_descifrada = None
        self._construir_ui()

    def _construir_ui(self):
        layout_principal = QVBoxLayout(self)
        titulo = QLabel("Métricas de Construcción (Fidelidad del Descifrado)")
        titulo.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 8px;")
        layout_principal.addWidget(titulo)

        info = QLabel("Compara la imagen original contra la imagen descifrada usando MSE, PSNR y SSIM.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; margin-bottom: 6px;")
        layout_principal.addWidget(info)

        barra_acciones = QHBoxLayout()
        self.btn_usar_sesion = QPushButton("↻ Usar últimas imágenes de la sesión")
        self.btn_usar_sesion.clicked.connect(self._usar_imagenes_de_sesion)
        barra_acciones.addWidget(self.btn_usar_sesion)

        self.btn_cargar_original = QPushButton("📁 Cargar Imagen Original")
        self.btn_cargar_original.clicked.connect(self._cargar_original)
        barra_acciones.addWidget(self.btn_cargar_original)

        self.btn_cargar_descifrada = QPushButton("📁 Cargar Imagen Descifrada")
        self.btn_cargar_descifrada.clicked.connect(self._cargar_descifrada)
        barra_acciones.addWidget(self.btn_cargar_descifrada)

        self.btn_calcular = QPushButton("📊 Calcular Métricas")
        self.btn_calcular.clicked.connect(self._calcular)
        barra_acciones.addWidget(self.btn_calcular)
        barra_acciones.addStretch()
        layout_principal.addLayout(barra_acciones)

        area_imagenes = QHBoxLayout()
        grupo_original = QGroupBox("Imagen Original")
        v_original = QVBoxLayout(grupo_original)
        self.label_original = QLabel("Ninguna imagen cargada")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setMinimumSize(300, 300)
        self.label_original.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_original.addWidget(self.label_original)
        area_imagenes.addWidget(grupo_original)

        grupo_descifrada = QGroupBox("Imagen Descifrada")
        v_descifrada = QVBoxLayout(grupo_descifrada)
        self.label_descifrada = QLabel("Ninguna imagen cargada")
        self.label_descifrada.setAlignment(Qt.AlignCenter)
        self.label_descifrada.setMinimumSize(300, 300)
        self.label_descifrada.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_descifrada.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_descifrada.addWidget(self.label_descifrada)
        area_imagenes.addWidget(grupo_descifrada)

        grupo_diff = QGroupBox("Mapa de Diferencias")
        v_diff = QVBoxLayout(grupo_diff)
        self.label_diff = QLabel("—")
        self.label_diff.setAlignment(Qt.AlignCenter)
        self.label_diff.setMinimumSize(300, 300)
        self.label_diff.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.label_diff.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_diff.addWidget(self.label_diff)
        area_imagenes.addWidget(grupo_diff)
        layout_principal.addLayout(area_imagenes, stretch=1)

        grupo_resultados = QGroupBox("Resultados")
        grid = QGridLayout(grupo_resultados)
        self.tarjeta_mse = TarjetaMetrica("MSE (Error Cuadrático Medio)")
        self.tarjeta_psnr = TarjetaMetrica("PSNR (dB)")
        self.tarjeta_ssim = TarjetaMetrica("SSIM")
        grid.addWidget(self.tarjeta_mse, 0, 0)
        grid.addWidget(self.tarjeta_psnr, 0, 1)
        grid.addWidget(self.tarjeta_ssim, 0, 2)
        layout_principal.addWidget(grupo_resultados)

        self.label_veredicto = QLabel("")
        self.label_veredicto.setAlignment(Qt.AlignCenter)
        self.label_veredicto.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 6px;")
        layout_principal.addWidget(self.label_veredicto)
        self._pixmap_original = None
        self._pixmap_descifrada = None
        self._pixmap_diff = None

    def _mostrar_original(self, imagen):
        self._imagen_original = imagen
        self._pixmap_original = pil_a_pixmap(imagen)
        self.label_original.setPixmap(escalar_pixmap_para_label(self._pixmap_original, self.label_original))

    def _mostrar_descifrada(self, imagen):
        self._imagen_descifrada = imagen
        self._pixmap_descifrada = pil_a_pixmap(imagen)
        self.label_descifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_descifrada, self.label_descifrada))

    def _usar_imagenes_de_sesion(self):
        original = estado_global.cifrar_imagen_original or estado_global.descifrar_imagen_final
        descifrada = estado_global.descifrar_imagen_final
        if original is None or descifrada is None:
            QMessageBox.information(self, "Sin datos de sesión", "Aún no hay un par original/descifrada disponible en esta sesión.")
            return
        self._mostrar_original(original)
        self._mostrar_descifrada(descifrada)
        QMessageBox.information(self, "Listo", "Imágenes de la sesión cargadas. Ahora presione 'Calcular Métricas'.")

    def _cargar_original(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen original", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: return
        try:
            self._mostrar_original(Image.open(ruta).convert("RGB"))
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _cargar_descifrada(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen descifrada", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: return
        try:
            self._mostrar_descifrada(Image.open(ruta).convert("RGB"))
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _calcular(self):
        if self._imagen_original is None or self._imagen_descifrada is None:
            QMessageBox.warning(self, "Faltan imágenes", "Debe cargar tanto la imagen original como la descifrada.")
            return

        resultado = evaluar_descifrado(self._imagen_original, self._imagen_descifrada)
        if "error" in resultado:
            QMessageBox.critical(self, "Error al calcular métricas", resultado["error"])
            return

        self.tarjeta_mse.set_valor(f"{resultado['mse']:.6f}")
        self.tarjeta_psnr.set_valor("∞ dB" if resultado["psnr"] == float("inf") else f"{resultado['psnr']:.2f} dB")
        self.tarjeta_ssim.set_valor(f"{resultado['ssim']:.4f}")
        self.label_veredicto.setText(resultado["veredicto"])
        
        color = "#2e7d32" if resultado["son_identicas"] else ("#ef6c00" if resultado["mse"] < 1.0 and resultado["ssim"] > 0.999 else "#c62828")
        self.label_veredicto.setStyleSheet(f"font-size: 13px; font-weight: bold; margin-top: 6px; color: {color};")

        try:
            mapa_diff = calcular_mapa_diferencias(self._imagen_original, self._imagen_descifrada)
            self._pixmap_diff = pil_a_pixmap(Image.fromarray(mapa_diff))
            self.label_diff.setPixmap(escalar_pixmap_para_label(self._pixmap_diff, self.label_diff))
        except Exception:
            pass

    def actualizar_al_mostrar(self):
        if self._pixmap_original is not None: self.label_original.setPixmap(escalar_pixmap_para_label(self._pixmap_original, self.label_original))
        if self._pixmap_descifrada is not None: self.label_descifrada.setPixmap(escalar_pixmap_para_label(self._pixmap_descifrada, self.label_descifrada))
        if self._pixmap_diff is not None: self.label_diff.setPixmap(escalar_pixmap_para_label(self._pixmap_diff, self.label_diff))


# ============================================================
#    PESTAÑA "METRICA DE ENTROPIA"
# ============================================================

class TabMetricaEntropia(QWidget):
    def __init__(self):
        super().__init__()
        self._imagen_original = None
        self._imagen_cifrada = None
        self._construir_ui()

    def _construir_ui(self):
        layout_principal = QVBoxLayout(self)
        titulo = QLabel("Métricas de Entropía y Correlación")
        titulo.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 8px;")
        layout_principal.addWidget(titulo)

        info = QLabel("Compara la imagen original (con padding) contra la imagen cifrada: histogramas, entropía de Shannon y correlación.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; margin-bottom: 6px;")
        layout_principal.addWidget(info)

        barra_acciones = QHBoxLayout()
        self.btn_usar_sesion = QPushButton("↻ Usar última imagen cifrada de la sesión")
        self.btn_usar_sesion.clicked.connect(self._usar_imagenes_de_sesion)
        barra_acciones.addWidget(self.btn_usar_sesion)

        self.btn_cargar_original = QPushButton("📁 Cargar Imagen Original")
        self.btn_cargar_original.clicked.connect(self._cargar_original)
        barra_acciones.addWidget(self.btn_cargar_original)

        self.btn_cargar_cifrada = QPushButton("📁 Cargar Imagen Cifrada")
        self.btn_cargar_cifrada.clicked.connect(self._cargar_cifrada)
        barra_acciones.addWidget(self.btn_cargar_cifrada)

        self.btn_calcular = QPushButton("📊 Calcular Métricas")
        self.btn_calcular.clicked.connect(self._calcular)
        barra_acciones.addWidget(self.btn_calcular)
        barra_acciones.addStretch()
        layout_principal.addLayout(barra_acciones)

        self.lbl_estado_imagenes = QLabel("Original: — | Cifrada: —")
        self.lbl_estado_imagenes.setStyleSheet("color: #555; margin-bottom: 4px;")
        layout_principal.addWidget(self.lbl_estado_imagenes)

        self.subtabs = QTabWidget()
        layout_principal.addWidget(self.subtabs, stretch=1)

        self.figura_hist = Figure(figsize=(11, 8))
        self.canvas_hist = FigureCanvas(self.figura_hist)
        scroll_hist = QScrollArea()
        scroll_hist.setWidgetResizable(True)
        scroll_hist.setWidget(self.canvas_hist)
        self.subtabs.addTab(scroll_hist, "Histogramas")

        self.figura_corr = Figure(figsize=(13, 9))
        self.canvas_corr = FigureCanvas(self.figura_corr)
        scroll_corr = QScrollArea()
        scroll_corr.setWidgetResizable(True)
        scroll_corr.setWidget(self.canvas_corr)
        self.subtabs.addTab(scroll_corr, "Correlación")

        widget_tablas = QWidget()
        layout_tablas = QVBoxLayout(widget_tablas)
        lbl_entropia = QLabel("Entropía de Shannon (bits)")
        lbl_entropia.setStyleSheet("font-weight: bold; margin-top: 4px;")
        layout_tablas.addWidget(lbl_entropia)

        self.tabla_entropia = QTableWidget(3, 3)
        self.tabla_entropia.setHorizontalHeaderLabels(["Canal", "Entropía Original", "Entropía Cifrada"])
        self.tabla_entropia.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_entropia.verticalHeader().setVisible(False)
        layout_tablas.addWidget(self.tabla_entropia)

        lbl_correlacion = QLabel("Coeficientes de Correlación de Píxeles Adyacentes")
        lbl_correlacion.setStyleSheet("font-weight: bold; margin-top: 12px;")
        layout_tablas.addWidget(lbl_correlacion)

        self.tabla_correlacion = QTableWidget(6, 4)
        self.tabla_correlacion.setHorizontalHeaderLabels(["Imagen / Dirección", "R", "G", "B"])
        self.tabla_correlacion.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_correlacion.verticalHeader().setVisible(False)
        layout_tablas.addWidget(self.tabla_correlacion)
        layout_tablas.addStretch()
        self.subtabs.addTab(widget_tablas, "Tablas de Valores")

    def _actualizar_estado_label(self):
        o = "cargada" if self._imagen_original is not None else "—"
        c = "cargada" if self._imagen_cifrada is not None else "—"
        self.lbl_estado_imagenes.setText(f"Original: {o} | Cifrada: {c}")

    def _usar_imagenes_de_sesion(self):
        original = estado_global.cifrar_imagen_padding or estado_global.cifrar_imagen_original
        cifrada = estado_global.cifrar_imagen_cifrada or estado_global.descifrar_imagen_cifrada
        if original is None or cifrada is None:
            QMessageBox.information(self, "Sin datos de sesión", "Aún no hay una imagen cifrada disponible en esta sesión.")
            return
        self._imagen_original = original
        self._imagen_cifrada = cifrada
        self._actualizar_estado_label()
        QMessageBox.information(self, "Listo", "Imágenes de la sesión cargadas. Ahora presione 'Calcular Métricas'.")

    def _cargar_original(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen original", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: return
        try:
            self._imagen_original = Image.open(ruta).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
        self._actualizar_estado_label()

    def _cargar_cifrada(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen cifrada", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not ruta: return
        try:
            self._imagen_cifrada = Image.open(ruta).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
        self._actualizar_estado_label()

    def _calcular(self):
        if self._imagen_original is None or self._imagen_cifrada is None:
            QMessageBox.warning(self, "Faltan imágenes", "Debe cargar tanto la imagen original como la cifrada.")
            return

        if self._imagen_original.size != self._imagen_cifrada.size:
            QMessageBox.warning(self, "Dimensiones distintas", f"Las imágenes tienen tamaños distintos ({self._imagen_original.size} vs {self._imagen_cifrada.size}).")

        datos_entropia = analizar_histogramas_entropia(self._imagen_original, self._imagen_cifrada)
        self._dibujar_histogramas(datos_entropia)
        self._llenar_tabla_entropia(datos_entropia)

        datos_correlacion = analizar_correlacion(self._imagen_original, self._imagen_cifrada)
        self._dibujar_correlacion(datos_correlacion)
        self._llenar_tabla_correlacion(datos_correlacion)

    def _dibujar_histogramas(self, datos):
        self.figura_hist.clear()
        canales, colores = datos["canales"], datos["colores"]
        axs = self.figura_hist.subplots(3, 2)
        self.figura_hist.suptitle("Análisis Estadístico: Distribución de Histogramas", fontsize=13, fontweight="bold")

        for i in range(3):
            hist_o, hist_c = datos["hist_original"][i], datos["hist_cifrada"][i]
            axs[i, 0].fill_between(range(256), hist_o, color=colores[i], alpha=0.85, edgecolor=colores[i], linewidth=0.6)
            axs[i, 0].set_title(f"{canales[i]} - Original", fontsize=10, fontweight="semibold")
            axs[i, 0].set_ylabel("Frecuencia (Píxeles)", fontsize=9)
            axs[i, 0].set_xlim([0, 255])
            axs[i, 0].grid(True, linestyle=":", alpha=0.6)

            axs[i, 1].fill_between(range(256), hist_c, color=colores[i], alpha=0.85, edgecolor=colores[i], linewidth=0.6)
            axs[i, 1].set_title(f"{canales[i]} - Cifrada", fontsize=10, fontweight="semibold")
            axs[i, 1].set_xlim([0, 255])
            axs[i, 1].grid(True, linestyle=":", alpha=0.6)

        axs[2, 0].set_xlabel("Intensidad del Píxel (0-255)", fontsize=9)
        axs[2, 1].set_xlabel("Intensidad del Píxel (0-255)", fontsize=9)
        self.figura_hist.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas_hist.draw()

    def _llenar_tabla_entropia(self, datos):
        for i in range(3):
            self.tabla_entropia.setItem(i, 0, QTableWidgetItem(datos["canales"][i]))
            self.tabla_entropia.setItem(i, 1, QTableWidgetItem(f"{datos['entropia_original'][i]:.6f}"))
            self.tabla_entropia.setItem(i, 2, QTableWidgetItem(f"{datos['entropia_cifrada'][i]:.6f}"))

    def _dibujar_correlacion(self, datos):
        self.figura_corr.clear()
        direcciones, canales, colores = datos["direcciones"], datos["canales"], datos["colores"]
        axs = self.figura_corr.subplots(2, 3)
        self.figura_corr.suptitle("Análisis de Correlación de Píxeles Adyacentes (RGB)", fontsize=13, fontweight="bold")

        for row, etiqueta in enumerate(["Original", "Encrypted"]):
            for col, dir_name in enumerate(direcciones):
                ax = axs[row, col]
                texto_leyenda = []
                for c_idx, letra in enumerate(canales):
                    coef, x_p, y_p = datos["datos"][etiqueta][dir_name][letra]
                    ax.scatter(x_p, y_p, s=2, color=colores[c_idx], alpha=0.4)
                    texto_leyenda.append(f"{letra}: {coef:.4f}")
                ax.set_title(f"{dir_name} ({etiqueta})", fontsize=9, fontweight="semibold")
                ax.set_xlim([0, 255])
                ax.set_ylim([0, 255])
                ax.set_aspect("equal", "box")
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.text(10, 185, "\n".join(texto_leyenda), fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

        self.figura_corr.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas_corr.draw()

    def _llenar_tabla_correlacion(self, datos):
        tabla, direcciones = datos["tabla_coeficientes"], datos["direcciones"]
        fila = 0
        for etiqueta in ["Original", "Encrypted"]:
            for dir_name in direcciones:
                self.tabla_correlacion.setItem(fila, 0, QTableWidgetItem(f"{etiqueta} - {dir_name}"))
                for col, letra in enumerate(["R", "G", "B"], start=1):
                    self.tabla_correlacion.setItem(fila, col, QTableWidgetItem(f"{tabla[etiqueta][dir_name][letra]:.4f}"))
                fila += 1

    def actualizar_al_mostrar(self): pass


# ============================================================
#    VENTANA PRINCIPAL 
# ============================================================

OPCIONES_SIDEBAR = [
    ("🔒  Cifrar", "cifrar"),
    ("🔓  Descifrar", "descifrar"),
    ("📈  Métrica de Construcción", "metrica_construccion"),
    ("🌀  Métrica de Entropía", "metrica_entropia"),
]


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cifrado de Imágenes — Arnold · Tent · Logístico Bidireccional")
        self.resize(1280, 820)
        self._worker_claves = None
        self._construir_ui()
        self._iniciar_carga_de_claves()

    def _construir_ui(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_raiz = QHBoxLayout(widget_central)
        layout_raiz.setContentsMargins(0, 0, 0, 0)
        layout_raiz.setSpacing(0)

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(240)
        self.sidebar.setIconSize(QSize(20, 20))
        self.sidebar.setStyleSheet(
            """QListWidget { background-color: #1f2733; border: none; padding-top: 10px; outline: 0; }
               QListWidget::item { color: #e6e6e6; padding: 14px 18px; font-size: 14px; }
               QListWidget::item:selected { background-color: #2f6fed; color: white; }
               QListWidget::item:hover:!selected { background-color: #2b3646; }"""
        )
        for texto, _clave in OPCIONES_SIDEBAR:
            self.sidebar.addItem(QListWidgetItem(texto))
        self.sidebar.currentRowChanged.connect(self._cambiar_pagina)
        layout_raiz.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #ffffff;")
        layout_raiz.addWidget(self.stack, stretch=1)

        self.tab_cifrar = TabCifrar(self.obtener_claves)
        self.tab_descifrar = TabDescifrar(self.obtener_claves)
        self.tab_metrica_construccion = TabMetricaConstruccion()
        self.tab_metrica_entropia = TabMetricaEntropia()

        self._paginas = [self.tab_cifrar, self.tab_descifrar, self.tab_metrica_construccion, self.tab_metrica_entropia]
        for pagina in self._paginas:
            self.stack.addWidget(pagina)
        self.sidebar.setCurrentRow(0)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_estado_claves = QLabel("Inicializando claves criptográficas...")
        self.status.addWidget(self.lbl_estado_claves)

    def _cambiar_pagina(self, indice):
        self.stack.setCurrentIndex(indice)
        pagina = self._paginas[indice]
        if hasattr(pagina, "actualizar_al_mostrar"):
            pagina.actualizar_al_mostrar()

    def obtener_claves(self): return estado_global.claves

    def _iniciar_carga_de_claves(self):
        self._deshabilitar_acciones_pesadas(True)
        self._worker_claves = WorkerGenerico(self._cargar_claves_y_precompilar)
        self._worker_claves.progreso.connect(self.lbl_estado_claves.setText)
        self._worker_claves.terminado.connect(self._al_terminar_carga_claves)
        self._worker_claves.error.connect(self._al_fallar_carga_claves)
        self._worker_claves.start()

    @staticmethod
    def _cargar_claves_y_precompilar(log=None):
        if log: log("Cargando claves criptográficas (Diffie-Hellman / HKDF)...")
        claves = obtener_claves()
        if log: log("Precompilando funciones criptográficas (Numba JIT)...")
        precompilar_jit()
        return claves

    def _al_terminar_carga_claves(self, claves):
        estado_global.claves = claves
        if claves.get("recien_generadas"):
            self.lbl_estado_claves.setText(f"Claves generadas y guardadas en: {claves['ruta_archivo']}")
        else:
            self.lbl_estado_claves.setText(f"Claves cargadas desde: {claves['ruta_archivo']}")
        self._deshabilitar_acciones_pesadas(False)

    def _al_fallar_carga_claves(self, mensaje_error):
        self.lbl_estado_claves.setText(f"Error al preparar claves: {mensaje_error}")
        self._deshabilitar_acciones_pesadas(False)

    def _deshabilitar_acciones_pesadas(self, deshabilitar):
        self.tab_cifrar.btn_cifrar.setEnabled(not deshabilitar and estado_global.cifrar_imagen_original is not None)
        self.tab_descifrar.btn_descifrar.setEnabled(not deshabilitar and estado_global.descifrar_imagen_cifrada is not None)



def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("Cifrado de Imágenes")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()