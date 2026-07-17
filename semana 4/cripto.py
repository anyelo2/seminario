import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

from decimal import Decimal, getcontext
from numba import njit
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
import hashlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor 


getcontext().prec = 50

_DOS   = Decimal('2')
_UNO   = Decimal('1')
_MEDIO = Decimal('0.5')
_D256  = Decimal('256')



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

# ── Logistico
@njit
def _generar_seq_logistico_jit(x0, L):
    seq = np.empty(L, dtype=np.uint8)
    x = x0
    for i in range(L):
        x = 4.0 * x * (1.0 - x)
        seq[i] = int(x * 256) % 256
    return seq

# ── Logistico: cifrado encadenado ────────────────────────────
@njit
def _logistico_cifrado_jit(datos, seq):
    L = len(datos)
    res = np.empty(L, dtype=np.uint8)
    res[0] = datos[0] ^ seq[0]
    for i in range(1, L):
        res[i] = datos[i] ^ seq[i] ^ res[i - 1]
    return res

# ── Logistico: descifrado encadenado ─────────────────────────
@njit
def _logistico_descifrado_jit(datos, seq):
    L = len(datos)
    res = np.empty(L, dtype=np.uint8)
    res[0] = datos[0] ^ seq[0]
    for i in range(1, L):
        res[i] = datos[i] ^ seq[i] ^ datos[i - 1]
    return res

# ── Precompilar todas las funciones JIT  
def precompilar_jit():
    print("Precompilando funciones JIT (Numba)...")
    c = np.zeros((4, 4), dtype=np.uint8)
    d = np.zeros(48, dtype=np.uint8)
    s = np.zeros(48, dtype=np.uint8)
    _arnold_cifrado_jit(c, np.int64(1), np.int64(1), np.int64(1), np.int64(4))
    _arnold_descifrado_jit(c, np.int64(1), np.int64(1), np.int64(1), np.int64(4))
    _generar_seq_logistico_jit(np.float64(0.5), 48)
    _logistico_cifrado_jit(d, s)
    _logistico_descifrado_jit(d, s)
    print("Compilacion JIT completada.\n")




def derivar_clave(semilla, contexto):
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=contexto.encode(),
    )
    return hkdf.derive(semilla)

def agregar_padding_cuadrado(imagen):
    ancho, alto = imagen.size
    lado = max(ancho, alto)
    imagen_cuadrada = Image.new('RGB', (lado, lado), color='black')
    izquierda = (lado - ancho) // 2
    arriba    = (lado - alto)  // 2
    imagen_cuadrada.paste(imagen, (izquierda, arriba))
    print(f"  Original: {imagen.size} -> Cuadrada: {imagen_cuadrada.size}")
    return imagen_cuadrada, (ancho, alto, izquierda, arriba)

def eliminar_padding(imagen_cuadrada, info_padding):
    ancho_orig, alto_orig, izquierda, arriba = info_padding
    region = (izquierda, arriba, izquierda + ancho_orig, arriba + alto_orig)
    recortada = imagen_cuadrada.crop(region)
    print(f"  Recortando: {imagen_cuadrada.size} -> {recortada.size}")
    return recortada


# ============================================
# ETAPA 1: ARNOLD (permutacion) — Numba JIT


def extraer_parametros_arnold(clave_bytes, N):
    valor = int.from_bytes(clave_bytes, "big")
    p  = np.int64(((valor >> 224) & 0xFFFFFFFF) % (N - 1) + 1)
    q  = np.int64(((valor >> 192) & 0xFFFFFFFF) % (N - 1) + 1)
    it = np.int64(((valor >> 184) & 0xFF) % 10 + 1)
    return p, q, it

def cifrado_arnold(imagen, clave):
    N = imagen.size[0]
    p, q, it = extraer_parametros_arnold(clave, N)
    print(f"  Arnold: p={p}, q={q}, iter={it}  ")
    canales  = [np.array(c, dtype=np.uint8) for c in imagen.split()]
    cifrados = [_arnold_cifrado_jit(c, p, q, it, np.int64(N)) for c in canales]
    return Image.merge(imagen.mode, [Image.fromarray(c, "L") for c in cifrados])

def descifrado_arnold(imagen_cifrada, clave):
    N = imagen_cifrada.size[0]
    p, q, it = extraer_parametros_arnold(clave, N)
    print(f"  Arnold: p={p}, q={q}, iter={it}")
    canales = [np.array(c, dtype=np.uint8) for c in imagen_cifrada.split()]
    desc    = [_arnold_descifrado_jit(c, p, q, it, np.int64(N)) for c in canales]
    return Image.merge(imagen_cifrada.mode, [Image.fromarray(c, "L") for c in desc])


# ============================================
# ETAPA 2: TENT (sustitucion) 


def extraer_x0_tent(clave_bytes):
    return Decimal(int.from_bytes(clave_bytes, 'big')) / Decimal(2**256)

def tent_paso(x):
    return _DOS * x if x < _MEDIO else _DOS * (_UNO - x)

def generar_secuencia_tent(x0, longitud):
    x   = x0
    seq = np.empty(longitud, dtype=np.uint8)
    for i in range(longitud):
        x      = tent_paso(x)
        seq[i] = int(x * _D256) % 256
    return seq

def _procesar_canal_tent_puro(args):

    canal_flat, x0, longitud, H, W = args
    seg_seq = generar_secuencia_tent(x0, longitud)
    resultado = (canal_flat ^ seg_seq).astype(np.uint8)
    return resultado.reshape(H, W)

def cifrado_tent(imagen, clave):
    arr     = np.array(imagen, dtype=np.uint8)
    H, W    = arr.shape[:2]
    tam_canal = H * W

    x0_base = extraer_x0_tent(clave)
    print(f"  Tent: x0_base={float(x0_base):.10f}")

    # Perturbaciones controladas para evitar keystreams identicos en canales
    x0_r = x0_base
    x0_g = (x0_base + Decimal('1e-15')) % Decimal('1')
    x0_b = (x0_base + Decimal('2e-15')) % Decimal('1')

    canales_flat = [arr[:, :, i].flatten() for i in range(3)]
    x0_canales   = [x0_r, x0_g, x0_b]

    tareas = [
        (canales_flat[i], x0_canales[i], tam_canal, H, W) 
        for i in range(3)
    ]

    
    with ProcessPoolExecutor(max_workers=3) as executor:
        resultados = list(executor.map(_procesar_canal_tent_puro, tareas))

    arr_resultado = np.stack(resultados, axis=2)
    return Image.fromarray(arr_resultado, mode='RGB')

def descifrado_tent(imagen_cifrada, clave):
    return cifrado_tent(imagen_cifrada, clave)


# ============================================
# ETAPA 3: LOGISTICO (difusion) — Numba JIT


def extraer_x0_logistico(clave_bytes):
    valor = int.from_bytes(clave_bytes, 'big')
    x0    = valor / (2**256)
    if x0 <= 0:  x0 = 1e-15
    if x0 >= 1:  x0 = 1 - 1e-15
    return np.float64(x0)

def cifrado_logistico(imagen, clave):
    datos = np.array(imagen, dtype=np.uint8).flatten()
    L     = len(datos)
    x0    = extraer_x0_logistico(clave)
    print(f"  Logistico: x0={x0:.10f}")
    seq       = _generar_seq_logistico_jit(x0, L)
    resultado = _logistico_cifrado_jit(datos, seq)
    forma     = np.array(imagen).shape
    return Image.fromarray(resultado.reshape(forma), mode=imagen.mode)

def descifrado_logistico(imagen_cifrada, clave):
    datos = np.array(imagen_cifrada, dtype=np.uint8).flatten()
    L     = len(datos)
    x0    = extraer_x0_logistico(clave)
    seq   = _generar_seq_logistico_jit(x0, L)
    resultado = _logistico_descifrado_jit(datos, seq)
    forma     = np.array(imagen_cifrada).shape
    return Image.fromarray(resultado.reshape(forma), mode=imagen_cifrada.mode)



def cifrar_imagen_completa(imagen_original, c_perm, c_sust, c_dif):
    print("\n" + "="*50)
    print("CIFRADO: Arnold -> Tent -> Logistico")
    print("="*50)

    t0 = time.time()
    imagen_pad, info_padding = agregar_padding_cuadrado(imagen_original)
    t_pad = time.time() - t0
    print(f"  Padding:     {t_pad:.4f} s")
    
    # Guardar imagen con padding 
    imagen_pad.save("01_imagen_con_padding.png")
    print("  Guardada: 01_imagen_con_padding.png")

    print("\nEtapa 1 - Arnold (permutacion):")
    t0 = time.time()
    imagen_arnold = cifrado_arnold(imagen_pad, c_perm)
    t_arnold = time.time() - t0
    print(f"  Tiempo:      {t_arnold:.4f} s")
    
    # Guardar imagen despues de Arnold 
    imagen_arnold.save("02_imagen_arnold_intermedia.png")
    print("  Guardada (intermedia): 02_imagen_arnold_intermedia.png")

    print("\nEtapa 2 - Tent (sustitucion):")
    t0 = time.time()
    imagen_tent = cifrado_tent(imagen_arnold, c_sust)
    t_tent = time.time() - t0
    print(f"  Tiempo:      {t_tent:.4f} s")
    
    # Guardar imagen despues de Tent 
    imagen_tent.save("03_imagen_tent_intermedia.png")
    print("  Guardada (intermedia): 03_imagen_tent_intermedia.png")

    print("\nEtapa 3 - Logistico (difusion):")
    t0 = time.time()
    imagen_cifrada = cifrado_logistico(imagen_tent, c_dif)
    t_log = time.time() - t0
    print(f"  Tiempo:      {t_log:.4f} s")
    
    # Guardar imagen final cifrada 
    imagen_cifrada.save("04_imagen_cifrada_completa.png")
    print("  Guardada: 04_imagen_cifrada_completa.png")

    t_total = t_pad + t_arnold + t_tent + t_log
    print(f"\n  TOTAL CIFRADO: {t_total:.4f} s")

    return imagen_cifrada, imagen_pad, info_padding, t_total


def descifrar_imagen_completa(imagen_cifrada, info_padding, c_perm, c_sust, c_dif):
    print("\n" + "="*50)
    print("DESCIFRADO: Logistico^-1 -> Tent^-1 -> Arnold^-1")
    print("="*50)

    print("\nEtapa 1 - Logistico inverso:")
    t0 = time.time()
    imagen_desc_log = descifrado_logistico(imagen_cifrada, c_dif)
    t_log = time.time() - t0
    print(f"  Tiempo:      {t_log:.4f} s")
    
    # Guardar despues de Logistico inverso 
    imagen_desc_log.save("05_desc_logistico_intermedia.png")
    print("  Guardada (intermedia): 05_desc_logistico_intermedia.png")

    print("\nEtapa 2 - Tent inverso :")
    t0 = time.time()
    imagen_desc_tent = descifrado_tent(imagen_desc_log, c_sust)
    t_tent = time.time() - t0
    print(f"  Tiempo:      {t_tent:.4f} s")
    
    # Guardar despues de Tent inverso 
    imagen_desc_tent.save("06_desc_tent_intermedia.png")
    print("  Guardada (intermedia): 06_desc_tent_intermedia.png")

    print("\nEtapa 3 - Arnold inverso:")
    t0 = time.time()
    imagen_desc_arnold = descifrado_arnold(imagen_desc_tent, c_perm)
    t_arnold = time.time() - t0
    print(f"  Tiempo:      {t_arnold:.4f} s")
    
    # Guardar despues de Arnold inverso 
    imagen_desc_arnold.save("07_desc_arnold_con_padding.png")
    print("  Guardada: 07_desc_arnold_con_padding.png")

    t0 = time.time()
    imagen_final = eliminar_padding(imagen_desc_arnold, info_padding)
    t_pad = time.time() - t0
    
    # Guardar imagen final descifrada SIN padding 
    imagen_final.save("08_imagen_descifrada_final.png")
    print("  Guardada: 08_imagen_descifrada_final.png")

    t_total = t_log + t_tent + t_arnold + t_pad
    print(f"\n  TOTAL DESCIFRADO: {t_total:.4f} s")

    return imagen_final, imagen_desc_arnold, t_total

def mostrar_imagenes(original, con_padding, cifrada, desc_pad, desc_final):
    """Muestra SOLO las 5 imágenes principales del pipeline"""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titulos  = [
        "1. Original", 
        "2. Con Padding", 
        "3. Cifrada Completa", 
        "4. Descifrada (con pad)", 
        "5. Descifrada Final"
    ]
    imagenes = [original, con_padding, cifrada, desc_pad, desc_final]
    
    for idx, (ax, img, titulo) in enumerate(zip(axes, imagenes, titulos)):
        if img is not None:
            ax.imshow(img)
            ax.set_title(f"{titulo}\n{img.size}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("09_pipeline_5_imagenes.png", dpi=150, bbox_inches='tight')
    print("  Guardada: 09_pipeline_5_imagenes.png")
    plt.show()




if __name__ == "__main__":
    t_inicio = time.time()

    # Compilar JIT antes de procesar la imagen
    precompilar_jit()

    # --- CRYPTO 
    priv_A = ec.generate_private_key(ec.SECP256K1())
    priv_B = ec.generate_private_key(ec.SECP256K1())

    pub_A = priv_A.public_key()
    pub_B = priv_B.public_key()

    secreto  = priv_A.exchange(ec.ECDH(), pub_B)
    secreto1 = priv_B.exchange(ec.ECDH(), pub_A)

    if secreto == secreto1:
        print("Secretos compartidos iguales")
        print(f"Secreto: {secreto.hex()}\n")

    semilla_maestra = hashlib.sha256(secreto).digest()
    print(f"Semilla maestra: {semilla_maestra.hex()}\n")

    clave_permutacion = derivar_clave(semilla_maestra, "permutacion")
    clave_sustitucion = derivar_clave(semilla_maestra, "sustitucion")   
    clave_difusion    = derivar_clave(semilla_maestra, "difusion")

    print("Claves derivadas:")
    print(f"  Permutacion (Arnold):   {clave_permutacion.hex()[:32]}...")
    print(f"  Sustitucion (Tent) K2:  {clave_sustitucion.hex()[:32]}...")
    print(f"  Difusion (Logistico):   {clave_difusion.hex()[:32]}...\n")

    print("Cargando imagen...")
    try:
        imagen_original = Image.open("imagen4.jpg").convert("RGB")
    except FileNotFoundError:
        print("  No se encontro imagen, creando imagen de prueba...")
        imagen_original = Image.new('RGB', (594, 450), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(imagen_original)
        draw.rectangle([100, 100, 494, 350], fill='blue')
        draw.ellipse([200, 150, 394, 300], fill='red')
        imagen_original.save("imagen4.jpg")
    print(f"  Imagen cargada: {imagen_original.size}")
    
    # Guardar imagen original
    imagen_original.save("00_imagen_original.png")
    print("  Guardada: 00_imagen_original.png")

    
    imagen_cifrada, imagen_pad, info_padding, t_cifrado = cifrar_imagen_completa(
        imagen_original, clave_permutacion, clave_sustitucion, clave_difusion
    )

    imagen_final, imagen_desc_arnold, t_descifrado = descifrar_imagen_completa(
        imagen_cifrada, info_padding, clave_permutacion, clave_sustitucion, clave_difusion
    )

    # Verificacion
    print("\n" + "="*50)
    print("VERIFICACION DE INTEGRIDAD")
    print("="*50)
    arr_orig = np.array(imagen_original)
    arr_desc = np.array(imagen_final)

    if np.array_equal(arr_orig, arr_desc):
        print(" La imagen descifrada es IDENTICA a la original")
    else:
        diff = np.abs(arr_orig.astype(np.int16) - arr_desc.astype(np.int16))
        print(f" ERROR: diferencia maxima={np.max(diff)}, pixeles distintos={np.count_nonzero(diff)}")

    # Resumen Final
    t_total = time.time() - t_inicio
    print("\n" + "="*50)
    print("RESUMEN DE TIEMPOS")
    print("="*50)
    print(f"  Cifrado:    {t_cifrado:.4f} s")
    print(f"  Descifrado: {t_descifrado:.4f} s")
    print(f"  Total:      {t_total:.4f} s")
    print("="*50)
    
    print("\n" + "="*50)
    print("IMÁGENES GUARDADAS")
    print("="*50)
    print("  [PIPELINE ]")
    print("  00_imagen_original.png")
    print("  01_imagen_con_padding.png")
    print("  04_imagen_cifrada_completa.png")
    print("  07_desc_arnold_con_padding.png")
    print("  08_imagen_descifrada_final.png")
    print("  09_pipeline_5_imagenes.png")
    print("\n  [IMÁGENES INTERMEDIAS]")
    print("  02_imagen_arnold_intermedia.png    ← Etapa 1")
    print("  03_imagen_tent_intermedia.png      ← Etapa 2")
    print("  05_desc_logistico_intermedia.png")
    print("  06_desc_tent_intermedia.png")
    print("="*50)

    
    mostrar_imagenes(
        imagen_original, 
        imagen_pad, 
        imagen_cifrada, 
        imagen_desc_arnold, 
        imagen_final         
    )
    print("\nProceso completado.")