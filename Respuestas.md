# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---

**Descripción y Análisis:**

El análisis descriptivo realizado en `ejercicio1_descriptivo.py` examina el dataset de precios de vehículos (_CarPrice_Assignment.csv_) compuesto incialmente por 205 instancias y 24 atributos. El script genera gráficas individuales por atributo respecto a la variable objetivo `Price`. Se realizaron los siguientes análisis:

- Cálculo de correlaciones.
- Estadística descriptiva (_media_, _skewness_, _curtosis_).
- Identificación de atributos con baja correlación.

Preliminarmente se eliminaron las columnas `car_ID` (identificador único sin valor predictivo) y `CarName` (nombre del modelo que no aporta información directa sobre el precio). El dataset resultante contiene 23 atributos, incluyendo variables numéricas (_int64_, _float64_) y categóricas (_str_).

El análisis de correlaciones reveló que 10 atributos numéricos tienen correlación menor a 0.70 con `Price`, sugiriendo que podrían ser excluidos en análisis futuros.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

**Fuente del dataset:** El dataset proviene del archivo `CarPrice_Assignment.csv` ubicado en la carpeta `data/`. Este dataset es público y está en [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction).

**Variable objetivo (target):** La variable objetivo es `Price`, que representa el precio del coche en dólares.

**Por qué tiene sentido hacer regresión sobre Price:**

1. **Variable continua numérica:** Es apropiado para modelos de regresión que predicen valores numéricos.
2. **Relación con características del vehículo:** El precio de un coche depende de múltiples características técnicas (tamaño del motor, peso, potencia, eficiencia de combustible, tipo de carrocería, etc.), lo cual sugiere que un modelo de regresión puede capturar estas relaciones.
3. **Interés práctico:** Predecir el precio de coches tiene aplicaciones reales en el mercado automotriz para valoración de coches usados, análisis de mercado y fijación de precios.
4. **Correlación con múltiples atributos:** El análisis de correlación muestra que varios atributos numéricos tienen correlaciones significativas con Price (algunas >0.70), lo que indica que existe una relación lineal que puede ser modelada con regresión.

---

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

**Distribución de las principales variables numéricas:**

1. **Enginesize (Tamaño del motor):**
   - Distribución: Asimetría positiva (skewness ≈ 1.0), con cola hacia la derecha
   - Media: ≈126.9, Curtosis: ≈1.5
   - Outliers: Se observan algunos valores extremos en el rango superior (motores >300), que corresponden a vehículos de alto rendimiento
   - Decisión: No se eliminaron outliers en este análisis descriptivo

2. **Curbweight (Peso en vacío):**
   - Distribución: Aproximadamente normal con leve asimetría positiva
   - Media: ≈2555.6, Skewness ≈ 0.5, Curtosis ≈ 0.1
   - Outliers: Algunos valores en el rango superior (pesos >4000) correspondientes a vehículos grandes
   - Decisión: No se eliminaron outliers en este análisis descriptivo

3. **Horsepower (Potencia):**
   - Distribución: Asimetría positiva marcada (skewness ≈ 1.0), cola derecha
   - Media: ≈104.3, Curtosis ≈ 0.6
   - Outliers: Valores extremos en el rango superior (>250 hp) correspondientes a vehículos deportivos
   - Decisión: No se eliminaron outliers en este análisis descriptivo

4. **Carwidth (Ancho del vehículo):**
   - Distribución: Aproximadamente normal
   - Media: ≈65.9, Skewness ≈ 0.5
   - Outliers: Mínimos, algunos valores en extremos
   - Decisión: No se eliminaron outliers

**Decisión general sobre outliers:** No se eliminaron outliers por los siguientes motivos:

- Los outliers corresponden a coches reales (deportivos, de lujo, SUVs) que son parte natural de la distribución del mercado.
- Eliminarlos podría sesgar el modelo y perder información sobre segmentos importantes del mercado.

---

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

**Tres variables numéricas con mayor correlación (en valor absoluto) con Price:**

1. **Enginesize:** Correlación ≈ 0.874
2. **Curbweight:** Correlación ≈ 0.835
3. **Horsepower:** Correlación ≈ 0.808

**Interpretación:**

- Todas las correlaciones son positivas como se esperaba.
- Enginesize tiene la correlación más alta, lo que sugiere que el tamaño del motor es el predictor más fuerte del precio entre las variables numéricas.
- Estas tres variables explican una porción significativa de la variabilidad en el precio, lo cual justifica su uso en modelos de regresión

---

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

**Valores nulos en el dataset:**

- **Resultado:** No hay valores nulos en ninguna de las 23 columnas del dataset
- **Porcentaje:** 0% de valores nulos

**Tratamiento de valores nulos:**

- El análisis de `df.isnull().any()` confirmó que todas las columnas tienen valores en todas sus instancias.

---

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

**Descripción y Análisis del Preprocesamiento:**

El preprocesamiento para la regresión lineal simple consistió en los siguientes pasos:

1. **Eliminación de columnas irrelevantes**: Se eliminaron `car_ID` y `CarName`.

2. **Selección del atributo predictor**: Se seleccionó solo a `Enginesize`, ya que este atributo mostró la correlación más alta con `Price` (≈0.87), lo que indica una fuerte relación lineal entre el tamaño del motor y el precio del coche.

3. **Escalado del predictor**: Se aplicó `StandardScaler`. Esta técnica centra los datos en media 0 con desviación estándar 1, lo cual es apropiado para algoritmos de regresión lineal que asumen datos en escalas similares y mejora la estabilidad numérica del modelo.

4. **División Train-Test**: De acuerdo a las indicaciones del ejercicio. Se dividió el dataset en 80% para entrenamiento (164 muestras) y 20% para prueba (41 muestras) usando `random_state=42`.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

**Valores de métricas en el test set:**

- MAE: 2748.69
- RMSE: 3932.61
- R²: 0.8041

**Coeficientes del modelo:**

- Intercepto (β₀): 13305.12
- Pendiente (β₁): 6889.35
- Ecuación: Price = 13305.12 + 6889.35 × Enginesize

**Evaluación del modelo:**
El modelo funciona razonablemente bien para ser una regresión lineal simple. El R² de 0.8041 indica que el modelo explica el 80.41% de la variabilidad en el precio usando únicamente el tamaño del motor, lo cual es un resultado sólido considerando la simplicidad del modelo.

No hay evidencia clara de overfitting: las métricas de entrenamiento (R²=0.7507, MAE=2856.77, RMSE=3855.82) y prueba (R²=0.8041, MAE=2748.69, RMSE=3932.61) son similares, con un rendimiento ligeramente mejor en test, lo cual es inusual pero positivo y sugiere que el modelo generaliza bien.

El atributo más influyente es `Enginesize` con un coeficiente positivo de 6889.35, lo que indica que por cada unidad adicional de tamaño del motor (en la escala estandarizada), el precio aumenta en promedio \$6,889.35. Esto tiene sentido lógico: motores más grandes generalmente implican coches más potentes y costosos.

Sin embargo, el MAE de \$2,748.69 indica que el modelo tiene un error promedio de casi \$3,000 en las predicciones, lo cual es significativo en el contexto de precios de coches que van desde \$5,000 hasta \$45,000. Esto sugiere que aunque `Enginesize` es un predictor importante, el precio de los coches depende de muchos otros factores que no están siendo considerados en este modelo simple.

---

**Conclusiones sobre la información útil del Ejercicio 1:**

La información del Ejercicio 1 fue fundamental para el desarrollo de la regresión lineal simple por las siguientes razones:

1. **Selección del predictor**: El análisis de correlación en el Ejercicio 1 identificó que `Enginesize` tenía la correlación más alta con `Price` (≈0.87), lo que justificó su selección como único predictor para la regresión lineal simple. Sin este análisis, habría sido necesario probar múltiples atributos para encontrar el más adecuado.

2. **Validación de la relación lineal**: Los scatter plots del Ejercicio 1 mostraron una relación lineal clara entre `Enginesize` y `Price`, lo que validó que una regresión lineal simple era apropiada para modelar esta relación.

3. **Identificación de outliers**: Los gráficos del Ejercicio 1 permitieron identificar outliers en `Enginesize` y `Price`. Aunque no se eliminaron en este ejercicio, conocer su presencia ayuda a interpretar por qué el modelo puede tener errores de predicción más altos en ciertos rangos de valores, lo cual se refleja en los residuos.

4. **Justificación del preprocesamiento**: El Ejercicio 1 confirmó que no había valores nulos, lo que simplificó el preprocesamiento. También mostró que `Enginesize` tenía una distribución aproximadamente normal, lo que justificó el uso de `StandardScaler` para el escalado del predictor.

5. **Interpretación de resultados**: El conocimiento de las estadísticas descriptivas del Ejercicio 1 (*media*, *skewness*, *curtosis* de `Enginesize`) ayuda a interpretar el coeficiente de la regresión. Por ejemplo, saber que `Enginesize` tiene una media de aproximadamente 130 permite contextualizar el impacto del coeficiente de 6889.35 en términos reales del tamaño del motor.

6. **Comparación con regresión múltiple**: El Ejercicio 1 identificó que otros atributos como `Curbweight` y `Horsepower` también tenían correlaciones altas con `Price`. Esto sugiere que una regresión lineal múltiple podría mejorar el R² actual de 0.8041, proporcionando una dirección clara para mejoras futuras del modelo.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

**Descripción y Análisis:**

El Ejercicio 3 implementa una regresión lineal múltiple utilizando únicamente `NumPy`, sin emplear `scikit-learn` para el ajuste del modelo. La implementación utiliza la solución analítica de Mínimos Cuadrados Ordinarios (OLS) para calcular los coeficientes del modelo mediante la fórmula $β = (XᵀX)⁻¹ Xᵀy$.

El script genera datos falsos con semilla fija (*seed=42*), con 200 muestras y 3 features. Los coeficientes reales conocidos son $β₀=5$, $β₁=2$, $β₂=-1$, $β₃=0.5$. Se añade ruido gaussiano ($σ=1.5$) a la variable objetivo para simular datos reales.

El preprocesamiento incluye:

- División train/test (80% / 20%) sin mezcla aleatoria
- Adición de columna de unos a la matriz X para el término independiente (intercepto)
- Cálculo de coeficientes mediante inversión matricial

El modelo se evalúa con métricas MAE, RMSE y R² sobre el conjunto de test, y se genera un gráfico de Valores Reales vs. Valores Predichos para visualizar el rendimiento del modelo.

**Resultados obtenidos:**

- MAE = 1.1665
- RMSE = 1.4612
- R² = 0.6897

El R² de 0.6897 indica que el modelo explica aproximadamente el 69% de la variabilidad en los datos de test, lo cual es aceptable considerando que se usaron datos falsos con ruido y que el split no fue aleatorio, lo que puede afectar la representatividad del test set.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula $β = (XᵀX)⁻¹ Xᵀy$ y por qué es necesario añadir una columna de unos a la matriz $X$.

**Explicación de la fórmula $β = (XᵀX)⁻¹ Xᵀy$:**

Esta fórmula es la solución analítica de Mínimos Cuadrados Ordinarios (OLS) para regresiones lineales. Busca encontrar el vector de coeficientes $β$ que minimiza la suma de los errores al cuadrado entre los valores observados y predichos.

**Componentes de la fórmula:**

- $XᵀX$: Es el producto matricial entre la transpuesta de la matriz de features X y X misma.
- $(XᵀX)⁻¹$: Es la inversa de la matriz $XᵀX$.
- $Xᵀy$: Es el producto entre la transpuesta de X y el vector de valores objetivo y. Esto representa la covarianza entre cada feature y la variable objetivo.
- $(XᵀX)⁻¹ Xᵀy$: Al multiplicar la inversa de $XᵀX$ por $Xᵀy$, se obtiene el vector de coeficientes $β$ que minimiza el error cuadrático medio.

**Por qué es necesario añadir una columna de unos a la matriz $X$:**

La columna de unos es necesaria para incluir el término independiente o intercepto ($β₀$) en el modelo. Sin esta columna:

- El modelo solo podría pasar por el origen (0,0), ya que la ecuación sería $y = β₁x₁ + β₂x₂ + ... + βₙxₙ$
- Con la columna de unos, la ecuación se convierte en $y = β₀·1 + β₁x₁ + β₂x₂ + ... + βₙxₙ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ$
- Esto permite que el modelo tenga un *offset* o desplazamiento vertical, lo cual es crucial para ajustar modelos donde la relación entre $X$ e $y$ no pasa necesariamente por el origen
- En términos matriciales, añadir la columna de ones hace que el primer coeficiente $β₀$ se multiplique por 1 para todas las observaciones, proporcionando el intercepto

---

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado | Diferencia |
| --------- | ---------- | -------------- | ---------- |
| β₀        | 5.0        | 4.86499486     | -0.1350    |
| β₁        | 2.0        | 2.06361770     | +0.0636    |
| β₂        | -1.0       | -1.11703839    | -0.1170    |
| β₃        | 0.5        | 0.43851694     | -0.0615    |

**Análisis de la comparación:**
Los coeficientes ajustados están muy cercanos a los valores reales, con diferencias pequeñas (todas menores a 0.15 en valor absoluto). Esto demuestra que la implementación de la solución OLS desde cero funciona correctamente. Las pequeñas discrepancias se deben al ruido gaussiano añadido a los datos (σ=1.5) y al hecho de que el modelo se ajusta solo con el conjunto de entrenamiento (80% de los datos), lo que introduce variabilidad en la estimación de los coeficientes.

---

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

**Valores obtenidos:**

- **MAE** = 1.1665
- **RMSE** = 1.4612
- **R²** = 0.6897

**Comparación con valores de referencia del enunciado:**

- MAE referencia: $≈1.20 (±0.20)$ → **Obtenido: 1.1665** ✓ (dentro del rango esperado)
- RMSE referencia: $≈1.50 (±0.20)$ → **Obtenido: 1.4612** ✓ (dentro del rango esperado)
- R² referencia: $≈0.80 (±0.05)$ → **Obtenido: 0.6897** ✗ (fuera del rango esperado)

**Análisis:**
El MAE y RMSE están dentro de los rangos esperados, lo que indica que el error de predicción es apropiado. Sin embargo, el R² obtenido (0.6897) es menor que el valor de referencia ($≈$0.80). Esta discrepancia se debe principalmente a que el split *train/test* no se realiza aleatoriamente (sin mezcla), lo que puede resultar en un test *set* que no es completamente representativo de la distribución de datos completa. En este caso específico, el test *set* probablemente contiene observaciones con mayor variabilidad o ruido que reducen el R² calculado.

---

**Pregunta 3.4** — Compara los resultados con la regresión lineal simple del Ejercicio 2 y comprueba si el resultado es parecido. Explica qué ha sucedido.

**Comparación con Ejercicio 2 (Regresión Lineal Simple):**

**Ejercicio 2 (Regresión Lineal Simple con Enginesize):**

- MAE = 2748.69
- RMSE = 3932.61
- R² = 0.8041

**Ejercicio 3 (Regresión Lineal Múltiple con datos falsos):**

- MAE = 1.1665
- RMSE = 1.4612
- R² = 0.6897

**Análisis de las diferencias:**

Los resultados **no son comparables** entre sí debido a razones fundamentales:

1. **Diferentes datasets:**
   - Ejercicio 2 usa datos reales de precios de coches (dataset `CarPrice_Assignment.csv`)
   - Ejercicio 3 usa datos falsos generados con distribución normal y ruido gaussiano

2. **Diferentes escalas:**
   - Ejercicio 2 trabaja con precios en dólares (rango $≈$5,000-$45,000$)
   - Ejercicio 3 trabaja con valores falsos en escala estándar (rango mucho menor)

3. **Diferentes objetivos:**
   - Ejercicio 2 es una regresión lineal simple (1 predictor) aplicada a un problema real.
   - Ejercicio 3 es una implementación didáctica de regresión lineal múltiple (3 predictors) desde cero con `NumPy`.

4. **Diferentes contextos de evaluación:**
   - Ejercicio 2 usa split *train/test* aleatorio (`random_state=42`).
   - Ejercicio 3 usa split *train/test* sin mezcla (determinístico por orden).

**Conclusión:**
La comparación no tiene sentido porque son ejercicios con propósitos completamente diferentes. El Ejercicio 2 busca resolver un problema real de predicción de precios, mientras que el Ejercicio 3 busca demostrar la comprensión de la implementación matemática de la regresión lineal múltiple usando álgebra lineal.

---

## Ejercicio 4 — Series Temporales

---

**Descripción y Análisis:**

El Ejercicio 4 realiza un análisis completo de una serie temporal falsa generada con semilla fija (seed=42), cubriendo el periodo del 1 de enero de 2018 al 31 de diciembre de 2023 (2191 observaciones diarias). La serie se genera con componentes conocidos: tendencia lineal creciente, estacionalidad anual, ciclos de largo plazo y ruido gaussiano.

El análisis incluye:

1. **Visualización de la serie completa**: Gráfico temporal de los 6 años de datos
2. **Descomposición de la serie**: Separación en componentes de Tendencia, Estacionalidad, Residuo y la serie original usando el modelo aditivo con `period=365` días
3. **Análisis del residuo**: Evaluación de si el residuo se comporta como ruido ideal mediante:
   - Estadísticos descriptivos (*media*, *std*, *asimetría*, *curtosis*)
   - Test de normalidad *Jarque-Bera*
   - Test de estacionariedad *ADF* (Augmented Dickey-Fuller)
   - Gráficos *ACF* y *PACF*
   - Histograma con curva normal superpuesta

**Resultados del análisis del residuo:**

- Media: 0.127 (muy cercana a 0, ideal para ruido)
- Std: 3.222 (cercana al valor teórico de 3.5 usado en la generación)
- Asimetría: -0.051 (cercana a 0, indica distribución simétrica)
- Curtosis: -0.061 (cercana a 0, indica distribución similar a normal)
- Test Jarque-Bera: $p=0.577 > 0.05$ (no rechazamos normalidad)
- Test ADF: $p≈0$ (serie estacionaria)

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

**Sí, la serie presenta una tendencia lineal creciente.**

- **Tipo:** Tendencia lineal (monótona)
- **Dirección:** Creciente (positiva)
- **Magnitud aproximada:** La tendencia tiene una pendiente de 0.05 unidades por día, lo que significa que la serie aumenta aproximadamente 0.05 unidades cada día. En el periodo de 6 años (2191 días), esto representa un aumento total de aproximadamente 109.55 unidades (0.05 × 2191), desde un valor inicial de aproximadamente 50 hasta valores finales cercanos a 160.

La tendencia es claramente visible en el gráfico de la serie original como un patrón ascendente sostenido a lo largo de todo el periodo, y se confirma en el componente de tendencia de la descomposición.

---

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

**Sí, la serie presenta estacionalidad anual.**

- **Periodo aproximado:** 365 días (1 año)
- **Amplitud del patrón estacional:** Aproximadamente ±21 unidades

La estacionalidad se manifiesta como un patrón repetitivo que ocurre una vez por año. En el gráfico de la serie original se pueden observar oscilaciones regulares que se repiten anualmente. El componente estacional de la descomposición muestra claramente este patrón sinusoidal con periodo de 365 días.

La amplitud de aproximadamente $±21$ unidades proviene de la combinación de dos componentes sinusoidales en la generación de la serie: $15 × sin(2πt/365.25) + 6 × cos(4πt/365.25)$, lo que produce un patrón estacional con variaciones máximas de aproximadamente 21 unidades por encima y por debajo de la tendencia.

---

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

**Sí, se aprecian ciclos de largo plazo en la serie.**

- **Periodo del ciclo:** Aproximadamente 4 años (1461 días)
- **Amplitud del ciclo:** Aproximadamente ±8 unidades

**Diferencia con la tendencia:**

- La **tendencia** es un patrón monótono y continuo que muestra la dirección general de la serie a largo plazo (en este caso, creciente de forma lineal). No oscila, solo aumenta o disminuye consistentemente.
- Los **ciclos de largo plazo** son oscilaciones que se repiten periódicamente pero con periodos mucho más largos que la estacionalidad anual. En este caso, el ciclo de ~4 años produce oscilaciones que van y vienen, subiendo y bajando, superpuestas a la tendencia creciente.

En el gráfico de la serie original, los ciclos de largo plazo se manifiestan como variaciones más suaves y lentas que la estacionalidad anual. Mientras que la estacionalidad produce oscilaciones rápidas (cada año), el ciclo de $4$ años produce oscilaciones más amplias y lentas que se observan como "olas" que cubren varios años.

La diferencia fundamental es que la tendencia representa la dirección general (monótona), mientras que los ciclos representan oscilaciones periódicas que van y vienen alrededor de esa tendencia.

---

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (*p-value*) para justificar tu respuesta.

**Sí, el residuo se ajusta muy bien a un ruido ideal.**

**Estadísticos del residuo:**

- **Media:** 0.127 (muy cercana a 0, ideal para ruido blanco)
- **Desviación típica (Std):** 3.222 (cercana al valor teórico de 3.5 usado en la generación)
- **Asimetría:** -0.051 (cercana a 0, indica distribución simétrica)
- **Curtosis:** -0.061 (cercana a 0, indica distribución similar a normal)

**Test de normalidad Jarque-Bera:**

- **Estadístico:** 1.101
- **p-valor:** 0.577
- **Conclusión:** Como *p-valor* $0.577 > 0.05$, no rechazamos la hipótesis de normalidad. El residuo sigue una distribución normal.

**Test de estacionariedad ADF:**

- **p-valor ADF:** $≈0.000000$
- **Conclusión:** Como *p-valor* $≤ 0.05$, el residuo es estacionaria (no tiene tendencia ni dependencia temporal).

**Justificación:**
El residuo cumple con todas las características de un ruido ideal (ruido blanco gaussiano):

1. **Media ≈ 0:** 0.127 es estadísticamente insignificante, muy cercano al valor ideal de 0
2. **Varianza constante:** La desviación típica de 3.222 es consistente con el valor teórico de 3.5
3. **Distribución normal:** El test Jarque-Bera no rechaza la normalidad ($p=0.577$)
4. **Sin autocorrelación:** El gráfico ACF muestra que los lags no son significativos (cerca de 0), lo que indica independencia temporal
5. **Estacionariedad:** El test ADF confirma que el residuo es estacionario

Por lo tanto, el residuo se comporta como un ruido blanco gaussiano ideal, lo que indica que la descomposición ha capturado correctamente todos los componentes sistemáticos de la serie (tendencia, estacionalidad y ciclos), dejando solo el componente aleatorio.

---

_Fin del documento de respuestas_
