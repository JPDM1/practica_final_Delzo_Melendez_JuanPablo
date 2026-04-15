import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')# más elegante
import seaborn as sns
sns.set_theme(style="whitegrid")

#Para evitar warnings
import warnings
warnings.filterwarnings('ignore')

#Añadir color a la terminal
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

print("\n" + color.BOLD + color.UNDERLINE+ color.CYAN + "ANÁLISIS DESCRIPTIVO" + color.END)
ruta_archivo = os.path.join('data', 'CarPrice_Assignment.csv')
df = pd.read_csv(ruta_archivo)
print("\n"+ color.BOLD + color.PURPLE + "Muestra de datos:" + color.END)
print(df.sample(10))

#Eliminado car_id y CarName que no aportan información
df=df.drop(columns=['car_ID', 'CarName'])

#Poner en mayúscula el primer caracter.
df.columns=[attr.title() for attr in df.columns]

#Viendo el tipo de dato de cada atributo
print("\n"+ color.BOLD + color.PURPLE + "Tipos de datos de cada atributo:" + color.END)
for i,attr in enumerate(df.columns,start=1):
    if df[attr].dtype == "str":
        print(f'{i:>2}.{attr +":":<20}'+ color.YELLOW + 'str' + color.END)
    elif df[attr].dtype == "int64":
        print(f'{i:>2}.{attr +":":<20}'+ color.RED + 'int64' + color.END)
    else:
        print(f'{i:>2}.{attr +":":<20}'+ color.BLUE + 'float64' + color.END)
              
#Viendo si hay valores nulos en el DataFrame
print("\n"+ color.BOLD + color.PURPLE + "¿Hay valores nulos?" + color.END)
print(df.isnull().any())

def analysis(attr):
    '''
    Función que genera un análisis descriptivo de un atributo.
    '''
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
        if unique_count <= 20:
            # Usar colores por categoría si hay pocas variables discretas
            order = variables_counted_df["x"].tolist()
            unique_attrs = sorted(df[attr].unique())
            colors = sns.color_palette('viridis', n_colors=len(unique_attrs))
            color_map = {attr_val: colors[i] for i, attr_val in enumerate(unique_attrs)}
            
            sns.barplot(data=variables_counted_df, x="x", y="y", hue="x", palette=color_map, legend=False, order=order)
            ax1.set(xlabel="",ylabel="")
            ax2=plt.subplot(212,sharex=ax1)
            sns.boxplot(data=df, y='Price', x=attr, hue=attr, palette=color_map, legend=False, order=order)
        else:
            # No usar colores individuales si hay muchas variables discretas
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
        corr_text = f"Correlación: {round(corr, 2)}"
        ax2.text(0.02, 0.95, corr_text, transform=ax2.transAxes, 
                 fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Añadir información de la media/moda según el tipo de variable
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
        # Para variables numéricas, mostrar la media
        mean_value = df[attr].mean()
        stats_text = f"Media: {round(mean_value, 2)}"
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                 fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Guardar la gráfica
    plt.savefig(os.path.join('output', f'eje1_{attr}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(color.GREEN + color.BOLD + "\nGenerando las gráficas..." + color.END)
for attr in df.columns[:-1]:
    analysis(attr)
    print(f"{attr}:" + color.CYAN + " ✓" + color.END)
print(color.GREEN + color.BOLD + f"¡¡{len(df.columns[:-1])} gráficas generadas!!" + color.END)
    


