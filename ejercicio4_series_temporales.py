"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 4
Análisis y Descomposición de Series Temporales
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio trabajarás con una serie temporal sintética generada con
una semilla fija. Tendrás que:

  1. Visualizar la serie completa.
  2. Descomponerla en sus componentes: Tendencia, Estacionalidad y Residuo.
  3. Analizar cada componente y responder las preguntas del fichero
     Respuestas.md (sección Ejercicio 4).
  4. Evaluar si el ruido (residuo) se ajusta a un ruido ideal (gaussiano
     con media ≈ 0 y varianza constante).

LIBRERÍAS PERMITIDAS
--------------------
  - numpy, pandas
  - matplotlib, seaborn
  - statsmodels   (para seasonal_decompose y adfuller)
  - scipy.stats   (para el test de normalidad del ruido)

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej4_serie_original.png      → Gráfico de la serie completa
  - output/ej4_descomposicion.png      → Los 4 subgráficos de descomposición
  - output/ej4_acf_pacf.png           → Gráfico ACF y PACF del residuo
  - output/ej4_histograma_ruido.png   → Histograma + curva normal del residuo
  - output/ej4_analisis.txt            → Estadísticos numéricos del análisis

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from anejos import color

#Configuración de estilos para gráficos
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# GENERACIÓN DE LA SERIE TEMPORAL SINTÉTICA — NO MODIFICAR ESTE BLOQUE
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.

    La serie tiene:
      - Una tendencia lineal creciente.
      - Estacionalidad anual (periodo 365 días).
      - Ciclos de largo plazo (periodo ~4 años).
      - Ruido gaussiano.

    Parámetros
    ----------
    semilla : int — Semilla aleatoria para reproducibilidad (NO modificar)

    Retorna
    -------
    serie : pd.Series con índice DatetimeIndex diario (2018-01-01 → 2023-12-31)
    """
    rng = np.random.default_rng(semilla)

    # Índice temporal: 6 años de datos diarios
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    # --- Componentes ---
    # 1. Tendencia lineal
    tendencia = 0.05 * t + 50

    # 2. Estacionalidad anual (periodo = 365.25 días)
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)

    # 3. Ciclo de largo plazo (periodo ~ 4 años = 1461 días)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)

    # 4. Ruido gaussiano
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    # Serie completa (modelo aditivo)
    valores = tendencia + estacionalidad + ciclo + ruido

    serie = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
# TAREA 1 — Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera y guarda un gráfico de la serie temporal completa.

    Salida: output/ej4_serie_original.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal a visualizar

    Pistas
    ------
    - Usa fig, ax = plt.subplots(figsize=(14, 4))
    - Añade título, etiquetas de ejes y una cuadrícula suave
    - Guarda con plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches='tight')
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, linewidth=1, color='steelblue')
    ax.set_title('Serie Temporal Completa (2018-2023)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# TAREA 2 — Descomposición de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone la serie en Tendencia, Estacionalidad y Residuo usando
    statsmodels.tsa.seasonal.seasonal_decompose y guarda el gráfico.

    Salida: output/ej4_descomposicion.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal

    Retorna
    -------
    resultado : DecomposeResult — Objeto con atributos .trend, .seasonal, .resid

    Pistas
    ------
    - from statsmodels.tsa.seasonal import seasonal_decompose
    - Usa model='additive' y period=365
    - resultado.plot() genera los 4 subgráficos automáticamente
    - Guarda la figura con fig.savefig(...)
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    resultado = seasonal_decompose(serie, model='additive', period=365)
    fig = resultado.plot()
    fig.set_figheight(10)
    fig.set_figwidth(14)
    fig.suptitle('Descomposición de la Serie Temporal', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("output/ej4_descomposicion.png", dpi=150, bbox_inches='tight')
    plt.close()
    return resultado


# =============================================================================
# TAREA 3 — Análisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente de residuo para determinar si se parece
    a un ruido ideal (gaussiano, media ≈ 0, varianza constante, sin autocorr.).

    Genera:
      - output/ej4_acf_pacf.png          → ACF y PACF del residuo
      - output/ej4_histograma_ruido.png  → Histograma + curva normal ajustada
      - output/ej4_analisis.txt          → Estadísticos numéricos

    Parámetros
    ----------
    residuo : pd.Series — Componente de residuo de la descomposición

    Pistas para el análisis numérico
    ----------------------------------
    - Media:     residuo.mean()
    - Std:       residuo.std()
    - Asimetría: residuo.skew()       (ruido ideal ≈ 0)
    - Curtosis:  residuo.kurtosis()   (ruido ideal ≈ 0 en exceso)
    - Test de normalidad Shapiro-Wilk o Jarque-Bera:
        from scipy.stats import jarque_bera
        stat, p = jarque_bera(residuo.dropna())
        Si p > 0.05 → no rechazamos normalidad
    - ADF Test para verificar estacionariedad:
        from statsmodels.tsa.stattools import adfuller
        resultado_adf = adfuller(residuo.dropna())
        p_adf = resultado_adf[1]
    - ACF / PACF:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy.stats import jarque_bera, norm
    
    # Limpiar el residuo
    residuo_limpio = residuo.dropna()
    
    # Calcular estadísticos básicos
    media = residuo_limpio.mean()
    std = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis = residuo_limpio.kurtosis()
    
    # Test de normalidad Jarque-Bera
    jb_stat, jb_p = jarque_bera(residuo_limpio)
    
    # Test de estacionariedad ADF
    resultado_adf = adfuller(residuo_limpio)
    p_adf = resultado_adf[1]
    
    # Gráfico ACF y PACF del residuo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(residuo_limpio, ax=ax1, lags=50, alpha=0.05)
    ax1.set_title('ACF del Residuo', fontsize=12, fontweight='bold')
    plot_pacf(residuo_limpio, ax=ax2, lags=50, alpha=0.05)
    ax2.set_title('PACF del Residuo', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/ej4_acf_pacf.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Histograma del residuo con curva normal
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuo_limpio, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Curva normal teórica
    xmin, xmax = residuo_limpio.min(), residuo_limpio.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, media, std)
    ax.plot(x, p, 'r-', linewidth=2, label=f'Normal (μ={media:.3f}, σ={std:.3f})')
    
    ax.set_title('Histograma del Residuo vs. Distribución Normal', fontsize=14, fontweight='bold')
    ax.set_xlabel('Valor del Residuo', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ej4_histograma_ruido.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Guardar estadísticos
    with open("output/ej4_analisis.txt", "w", encoding='utf-8') as f:
        f.write("Analisis del Residuo (Ruido)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Estadisticos basicos:\n")
        f.write(f"  Media:     {media:.6f}\n")
        f.write(f"  Std:       {std:.6f}\n")
        f.write(f"  Asimetria: {asimetria:.6f}\n")
        f.write(f"  Curtosis:  {curtosis:.6f}\n\n")
        f.write("Test de normalidad Jarque-Bera:\n")
        f.write(f"  Estadistico: {jb_stat:.6f}\n")
        f.write(f"  p-valor:    {jb_p:.6f}\n")
        f.write(f"  Conclusion: {'No rechazamos normalidad (p > 0.05)' if jb_p > 0.05 else 'Rechazamos normalidad (p <= 0.05)'}\n\n")
        f.write("Test de estacionariedad ADF:\n")
        f.write(f"  p-valor ADF: {p_adf:.6f}\n")
        f.write(f"  Conclusion: {'La serie es estacionaria (p <= 0.05)' if p_adf <= 0.05 else 'La serie no es estacionaria (p > 0.05)'}\n")


# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print(color.BOLD + color.UNDERLINE + color.PURPLE + "EJERCICIO 4 — Análisis de Series Temporales" + color.END)
    print("=" * 55)

    # ------------------------------------------------------------------
    # Paso 1: Generar la serie (NO modificar la semilla)
    # ------------------------------------------------------------------
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)
    

    print("\n" + color.CYAN + "Serie generada:" + color.END)
    print(f"  Periodo:      {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # ------------------------------------------------------------------
    # Paso 2: Visualizar la serie completa
    # ------------------------------------------------------------------
    print("\n" + color.GREEN + "[1/3] Visualizando la serie original..." + color.END)
    visualizar_serie(serie)

    # ------------------------------------------------------------------
    # Paso 3: Descomponer
    # ------------------------------------------------------------------
    print(color.GREEN + "[2/3] Descomponiendo la serie..." + color.END)
    resultado = descomponer_serie(serie)

    # ------------------------------------------------------------------
    # Paso 4: Analizar el residuo
    # ------------------------------------------------------------------
    print(color.GREEN + "[3/3] Analizando el residuo..." + color.END)
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # ------------------------------------------------------------------
    # Resumen de salidas esperadas
    # ------------------------------------------------------------------
    print("\n" + color.GREEN + "Salidas esperadas en output/:" + color.END)
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(color.GREEN + f"[{estado}]" + color.END, f" output/{s}")

    print("\n" + color.YELLOW + "¡Recuerda completar las respuestas en Respuestas.md!" + color.END)
