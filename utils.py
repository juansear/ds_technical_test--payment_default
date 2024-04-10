import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as sm


def dar_estilo_visual():
    plt.rcParams['figure.dpi'] = 100
    sns.set(font_scale = 1)
    sns.set_style({'font.family':'STIXGeneral',
                   'axes.grid' : True,
                   'grid.color': '0.8',
                   'axes.facecolor':'white',
                   'saturation' : 1})
    sns.set_palette("husl", 8)
    return None


def reporte_desempeno(real, pred, titulo="", save_name_path=""):
    """ Función para generar la matriz de confusión y hallar las principales métricas de desempeño de un modelo de clasificación teniendo en cuenta
    un conjunto de predicciones y el valor real asociado a cada una de éstas 

    Args:
        real (np.array o lista): Vector binario con los valores reales de la variable objetivo codificados con ceros y unos
        pred (np.array o lista): Vector binario con los valores predichos por el modelo de clasificación, codificados con ceros y unos
        titulo (str, optional): Título del modelo que se está evaluando. Defaults to "".
        save_name_path (str, optional): Nombre y ruta en caso de querer guardar el gráfico, si no se otorga, no se guarda.
    """
    metricas = {
        "F1 Score":sm.f1_score, 
        "Accuracy":sm.accuracy_score, 
        "Recall":sm.recall_score, 
        "Precision":sm.precision_score,
        "ROC-AUC": sm.roc_auc_score,
        }

    ## Métricas en la parte superior
    # Calcular Métricas
    results =[ (nombre, "{:.3f}".format(metrica(real, pred))) for nombre, metrica in metricas.items()]
    # Crear figura y definir su tamaño
    fig, ax = plt.subplots(figsize=(7,6))
    # Agregar título en la esquina superior izquierda
    plt.figtext(0.1, 1.075, titulo) 
    # Agregar texto de las métricas en la parte superior derecha de la figura    
    table = ax.table(cellText=results, loc="top", bbox=[0.5,1.1,0.7,0.4])
    table.scale(1, 3)

    
    ## Matriz de confusión 
    # Hallar matriz de confusión
    matriz_conf = sm.confusion_matrix(real, pred)
    # Calcular Totales y Porcentajes en la matriz de confusion
    totales = ["{0:0.0f}".format(value) for value in matriz_conf.flatten()]
    porcentaje = ["{0:.2%}".format(value) for value in matriz_conf.flatten()/np.sum(matriz_conf)]
    # Texto en los recuardos internos
    etiquetas = [ f"{total}\n{porcentaje}" for total,porcentaje in zip(totales,porcentaje)  ]
    etiquetas = np.asarray(etiquetas).reshape(2,2)
    # Matriz de confusión como mapa de calor al estilo Seaborn
    ax = sns.heatmap(matriz_conf, annot=etiquetas, fmt='', cmap="Greens" )
    ax.set_xlabel(" \nPredicciones\n ", fontsize="large")
    ax.set_ylabel("\nReal\n ", fontsize="large")
    # Márgenes holgadas
    #plt.tight_layout()
    

    if len(save_name_path) > 0:
        plt.savefig(save_name_path, bbox_inches="tight")
        return None

    plt.show()

    return None



# Función para graficar la importancia de las características
def plot_feature_importance(importances, features, title):
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importancia')
    plt.ylabel('Variables')
    plt.tight_layout()
    plt.show()