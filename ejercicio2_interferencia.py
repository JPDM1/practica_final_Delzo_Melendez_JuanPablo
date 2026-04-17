"""
Ejercicio 2 - Regresión Lineal Simple
Entrenamiento y evaluación de modelo de regresión lineal simple
para predecir la variable objetivo Price usando un solo atributo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

#Ingnorar warnings
import warnings
warnings.filterwarnings('ignore')

#Importar colores para el print
from anejos import color

# Configuración de estilo para gráficos
plt.style.use('classic')
sns.set_theme(style="whitegrid")

if __name__ == "__main__":
    # Cargar datos
    ruta_archivo = os.path.join('data', 'CarPrice_Assignment.csv')
    df = pd.read_csv(ruta_archivo)

    print("\n" + color.BOLD + color.UNDERLINE + color.PURPLE + 
           "Ejercicio 2: REGRESIÓN LINEAL SIMPLE" + color.END)

    # Eliminar columnas que no aportan información relevante
    # car_ID es solo un identificador y CarName no aporta valor predictivo directo
    df = df.drop(columns=['car_ID', 'CarName'])

    # Poner en mayúscula el primer caracter de los nombres de columnas
    df.columns = [attr.title() for attr in df.columns]

    # ============================================================================
    # PREPROCESAMIENTO - REGRESIÓN LINEAL SIMPLE
    # ============================================================================

    print("\n" + color.BOLD + color.CYAN + "PREPROCESAMIENTO\n" + color.END)

    # Para regresión lineal simple, usamos solo un atributo con la más alta correlación con Price
    # Según en el anterior apartado, Enginesize tiene la correlación más alta con Price (≈0.87)
    feature_col = 'Enginesize'
    
    print(color.CYAN + "Atributo predictor seleccionado:" + color.END, feature_col)
    print(color.CYAN + "Variable objetivo:" + color.END, "Price")

    # Escalado del atributo predictor usando StandardScaler
    # StandardScaler centra los datos en 0 con desviación estándar 1
    print(color.GREEN + "Escalando atributo predictor con StandardScaler..." + color.END)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[[feature_col]])
    X = pd.DataFrame(X, columns=[feature_col])
    y = df.Price

    print("\n" + color.CYAN + "Dimensiones después del preprocesamiento:" + color.END)
    print(f"X (feature): {X.shape}")
    print(f"y (target): {y.shape}")

    # ============================================================================
    # DIVISIÓN TRAIN-TEST
    # ============================================================================

    print("\n" + color.BOLD + color.CYAN + "DIVISIÓN TRAIN-TEST" + color.END)

    # Dividir datos en Train (80%) y Test (20%) con random_state=42 para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(color.CYAN + "Conjunto de entrenamiento:" + color.END, f"{X_train.shape[0]} muestras (80%)")
    print(color.CYAN + "Conjunto de prueba:" + color.END, f"{X_test.shape[0]} muestras (20%)")

    # ============================================================================
    # MODELO DE REGRESIÓN LINEAL SIMPLE
    # ============================================================================

    print("\n" + color.BOLD + color.CYAN + "ENTRENAMIENTO DEL MODELO" + color.END)

    # Crear y entrenar el modelo de Regresión Lineal Simple
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(color.GREEN + "Modelo entrenado exitosamente" + color.END)

    # Mostrar coeficientes del modelo
    print("\n" + color.CYAN + "Coeficientes del modelo:" + color.END)
    print(f"  Intercepto (β₀): {model.intercept_:.2f}")
    print(f"  Pendiente (β₁): {model.coef_[0]:.2f}")
    print(f"\n  Ecuación: Price = {model.intercept_:.2f} + {model.coef_[0]:.2f} × {feature_col}")

    # ============================================================================
    # EVALUACIÓN DEL MODELO
    # ============================================================================

    print("\n" + color.BOLD + color.CYAN + "EVALUACIÓN DEL MODELO" + color.END)

    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + color.CYAN + "Métricas en conjunto de ENTRENAMIENTO:" + color.END)
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")

    print("\n" + color.CYAN + "Métricas en conjunto de PRUEBA (TEST):" + color.END)
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")

    # ============================================================================
    # GRÁFICO DE RESIDUOS
    # ============================================================================

    print("\n" + color.BOLD + color.CYAN + "GENERANDO GRÁFICO DE RESIDUOS" + color.END)

    # Calcular residuos
    residuos = y_test - y_test_pred

    # Crear gráfico de residuos
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_pred, residuos, alpha=0.6, edgecolors='k')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Valores Predichos', fontsize=12)
    plt.ylabel('Residuos', fontsize=12)
    plt.suptitle('Gráfico de Residuos: Predichos vs Residuos', fontsize=14, fontweight='bold', y=0.98)
    plt.title('Atributo predictor: ' + feature_col, fontsize=11, y=1.02)
    plt.grid(True, alpha=0.3)

    # Guardar gráfico
    plt.savefig(os.path.join('output', 'ej2_residuos.png'), dpi=300, bbox_inches='tight')
    print(color.GREEN + "Gráfico guardado en: output/ej2_residuos.png" + color.END)
    plt.close()

    # ============================================================================
    # GUARDAR MÉTRICAS
    # ============================================================================

    print("\n" + color.BOLD + color.CYAN + "GUARDANDO MÉTRICAS" + color.END)

    # Guardar métricas en archivo de texto
    metricas_text = f"""MÉTRICAS DE REGRESIÓN LINEAL SIMPLE (TEST SET)
    =============================================
    Atributo predictor: {feature_col}
    MAE:  {test_mae:.4f}
    RMSE: {test_rmse:.4f}
    R²:   {test_r2:.4f}
    """

    with open(os.path.join('output', 'ej2_metricas_regresion.txt'), 'w') as f:
        f.write(metricas_text)
    
    print(color.GREEN + "Métricas guardadas en: output/ej2_metricas_regresion.txt" + color.END)
    
    print("\n" + color.GREEN + color.BOLD + "¡¡PROCESO COMPLETADO!!\n" + color.END)
