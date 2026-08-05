# Seminario de Investigación
# Herramienta de cifrado de imágenes (GUI)
archivo de la herramienta llamada `.herramienta.py`


##  Características Principales

  Acuerdo de Claves Seguro:** Generación de claves mediante Diffie-Hellman de Curva Elíptica (ECDH con `secp256k1`) y derivación mediante HKDF-SHA256.
 Alto Rendimiento:** Funciones críticas de cifrado aceleradas con compilación JIT de **Numba** y procesamiento paralelo de canales de color con `ProcessPoolExecutor`.
 
 Análisis Criptográfico Integrado:** Herramientas incorporadas para calcular y visualizar:
  - **Fidelidad:** MSE, PSNR, SSIM y Mapa de Diferencias.
  - **Estadística:** Entropía de Shannon, Histogramas por canal (RGB).
  - **Correlación:** Coeficientes de correlación de píxeles adyacentes (Horizontal, Vertical, Diagonal).
  - **Gestión de Padding:** Relleno automático a cuadrado y guardado de metadatos (`.json`) para un recorte perfecto durante el descifrado.

---

## ⚙️ ¿Cómo funciona el Pipeline?

1. **Acuerdo de Claves:** Se simula un intercambio ECDH para generar una semilla maestra, de la cual se derivan 3 claves independientes (Permutación, Sustitución, Difusión).
2. **Etapa 1 - Permutación (Arnold):** Mezcla las posiciones de los píxeles espacialmente utilizando el Mapa del Gato de Arnold, con parámetros `p`, `q` e iteraciones derivadas de la clave.
3. **Etapa 2 - Sustitución (Tent):** Modifica los valores de los píxeles usando el Mapa Caótico de Tent. Se aplican micro-perturbaciones a la condición inicial `x0` para cada canal (R, G, B) para evitar patrones repetitivos.
4. **Etapa 3 - Difusión (Logístico Bidireccional):** Aplica 2 rondas de barrido hacia adelante y hacia atrás usando suma modular (`+`). Esto asegura que un cambio de 1 bit en la imagen original altere ~ los bits en la imagen cifrada (Efecto Avalancha).

---

## 📦 Instalación
para el uso de la herramienta con gui integrada instalar las siguientes dependencias usando el siguiente comando:
pip install numpy Pillow numba cryptography scikit-image matplotlib PyQt5
