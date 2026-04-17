"""
Ejercicio 1 - Análisis Descriptivo
Script que realiza análisis descriptivo del dataset de precios de coches,
generando gráficas individuales para cada atributo respecto a la variable objetivo Price,
calculando correlaciones, estadísticas descriptivas.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Configuración de estilos para gráficos
plt.style.use('classic')
sns.set_theme(style="whitegrid")

#Importar colores para el print
from anejos import color

#Para evitar warnings
import warnings
warnings.filterwarnings('ignore')


print("\n" + color.BOLD + color.UNDERLINE+ color.PURPLE + "Ejercicio 1: ANÁLISIS DESCRIPTIVO" + color.END)
ruta_archivo = os.path.join('data', 'CarPrice_Assignment.csv')
df = pd.read_csv(ruta_archivo)

# ============================================================================
# PREPROCESAMIENTO
# ============================================================================

print("\n" + color.BOLD + color.CYAN + "PREPROCESAMIENTO" + color.END)

#Eliminado car_id y CarName que no aportan información
df=df.drop(columns=['car_ID', 'CarName'])
print("Se eliminaron las columnas 'car_ID' y 'CarName'\n")
#Poner en mayúscula el primer caracter.
df.columns=[attr.title() for attr in df.columns]

print(color.BOLD + color.CYAN + "Dimensiones del dataframe:" + color.END, 
f"{df.shape[0]} instancias y {df.shape[1]} atributos")

#Dtypes
print("\n"+ color.BOLD + color.CYAN + "Tipo de variable por atributo:" + color.END)
for i,attr in enumerate(df.columns,start=1):
    if df[attr].dtype == "str":
        print(f'{i:>2}.{attr:<16}:'+ color.YELLOW + 'str' + color.END)
    elif df[attr].dtype == "int64":
        print(f'{i:>2}.{attr:<16}:'+ color.RED + 'int64' + color.END)
    else:
        print(f'{i:>2}.{attr:<16}:'+ color.BLUE + 'float64' + color.END)

#Muestra aleatoria
print("\n"+ color.BOLD + color.CYAN + "Muestra de datos:" + color.END)
print(df.sample(10))

#Viendo si hay valores nulos en el DataFrame
print("\n"+ color.BOLD + color.CYAN + "¿Hay valores nulos?" + color.END)
nulos = df.isnull().any()
for attr, tiene_nulos in nulos.items():
    if tiene_nulos:
        print(f"{attr:<16}: {color.RED + 'Sí' + color.END}")
    else:
        print(f"{attr:<16}: {color.BLUE + 'No' + color.END}")

# ============================================================================
# CREACIÓN DE GRÁFICAS
# ============================================================================

# Diccionario para almacenar correlaciones de atributos numéricos
correlaciones = {}

def analysis(attr):
    """
    Función que genera un análisis descriptivo por atributo respecto a la variable objetivo "Price".
    """
    #Contando los valores únicos y creando un DataFrame con ellos
    variables_counted=df[attr].value_counts(ascending=True).sort_index()
    variables_counted_df =pd.DataFrame({"x":variables_counted.index,"y":variables_counted.values})
    #Tipo de dato
    dtype= df[attr].dtype
    #Cantidad de valores únicos
    #unique_count = df[attr].nunique()
    unique_count = variables_counted_df.size
    #2 gráficos
    plt.figure(figsize=(18,10))
    ax1=plt.subplot(211)
    if (dtype== "float64") or (dtype=="int64" and variables_counted.size>30):
        sns.histplot(data=df[attr], bins=100)
        ax1.set(xlabel="",ylabel="")
        ax2=plt.subplot(212,sharex=ax1)
        sns.scatterplot(data=df, x=attr, y="Price")
    elif dtype=="str":
        order = variables_counted_df["x"].tolist()
        # Crear un mapeo de colores consistente
        unique_attrs = df[attr].unique()
        colors = sns.color_palette('viridis', n_colors=len(unique_attrs))
        color_map = {attr_val: colors[i] for i, attr_val in enumerate(sorted(unique_attrs))}
        
        sns.barplot(data=variables_counted_df, x="y", y="x", hue="x", palette=color_map, legend=False, order=order)
        ax1.set(xlabel="",ylabel="")
        ax2=plt.subplot(212)
        sns.boxplot(data=df, y='Price', x=attr, hue=attr, palette=color_map, legend=False, order=order)
    else:
        # Solo usar colores por categoría si hay pocas variables discretas
        if unique_count <= 20:
            order = variables_counted_df["x"].tolist()
            unique_attrs = sorted(df[attr].unique())
            colors = sns.color_palette('viridis', n_colors=len(unique_attrs))
            color_map = {attr_val: colors[i] for i, attr_val in enumerate(unique_attrs)}
            
            # Para variables numéricas, no usar hue para mantener alineación correcta
            sns.barplot(data=variables_counted_df, x="x", y="y", palette='viridis', order=order)
            ax1.set(xlabel="",ylabel="")
            ax2=plt.subplot(212,sharex=ax1)
            sns.boxplot(data=df, y='Price', x=attr, palette='viridis', order=order)
        else:
            sns.barplot(data=variables_counted_df, x="x", y="y", palette='viridis')
            ax1.set(xlabel="",ylabel="")
            ax2=plt.subplot(212,sharex=ax1)
            sns.boxplot(data=df, y='Price', x=attr, palette='viridis')
    ax2.set(xlabel="")
    plt.suptitle(attr,fontsize=18,fontweight="bold")
    
    # Añadir número de correlación dentro del gráfico
    if dtype != 'str':
        df_temp = df[[attr,'Price']].copy()
        corr=df_temp.corr().iloc[0,1]
        # Guardar correlación en el diccionario
        correlaciones[attr] = corr
        corr_text = f"Correlación: {round(corr, 2)}"
        ax2.text(0.02, 0.95, corr_text, transform=ax2.transAxes, 
                 fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Añadir información según el tipo de variable
    if dtype == 'str':
        # Para variables categóricas, mostrar la moda (valor más frecuente)
        mode_value = df[attr].mode()[0]
        mode_count = df[attr].value_counts()[mode_value]
        # Mostrar en dos líneas para mejor legibilidad
        stats_text = f"Moda: {mode_value}\nFrecuencia: {mode_count}"
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                 fontsize=13, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        # Para variables numéricas: media, skewness y curtosis
        mean_value = df[attr].mean()
        skewness = df[attr].skew()
        kurtosis = df[attr].kurtosis()
        stats_text = f"Media: {round(mean_value, 2)}\nSkewness: {round(skewness, 2)}\nCurtosis: {round(kurtosis, 2)}"
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                 fontsize=13, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Guardar la gráfica
    plt.savefig(os.path.join('output', f'ej1_{attr}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(color.CYAN + color.BOLD + "\nGENERANDO LAS GRÁFICAS" + color.END)
print(color.GREEN + "Guardando en output/" + color.END)
for attr in df.columns[:-1]:
    analysis(attr)
    print(f"{attr}:" + color.GREEN + " ✓" + color.END)
print(color.GREEN + color.BOLD + f"¡¡{len(df.columns[:-1])} gráficas generadas!!\n" + color.END)

# ============================================================================
# ANÁLISIS DE CORRELACIONES
# ============================================================================

print(color.BOLD + color.CYAN + "ANÁLISIS DE CORRELACIONES CON PRICE" + color.END)
print(color.CYAN + "Atributos numéricos con correlación < 0.70:" + color.END)

atributos_baja_corr = []
for attr, corr in correlaciones.items():
    if abs(corr) < 0.70:
        atributos_baja_corr.append((attr, corr))

# Ordenar por valor absoluto de correlación (de menor a mayor)
atributos_baja_corr.sort(key=lambda x: abs(x[1]))

for attr, corr in atributos_baja_corr:
    print(  f'{attr:<16}:' , color.RED + f'{str(round(corr, 3)):>1}' + color.END)

if not atributos_baja_corr:
    print(color.GREEN + "  Ningún atributo numérico tiene correlación < 0.70" + color.END)
else:
    print(color.YELLOW + f"\nSe recomienda considerar la exclusión de estos {len(atributos_baja_corr)} atributos" + color.END)

print()
    


