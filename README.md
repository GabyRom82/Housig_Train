<h1 align="center"> DATA ANALYST portfolio (predicción de precios) </h1> 
<p align="center">
  <img width="275" height="183" alt="image" src="https://github.com/user-attachments/assets/889be6d9-db76-4e55-8f02-55a19b04b840" />
</p>

# DESCRIPCIÓN DEL PROYECTO
   ## Predicción y Análisis de Precios de Casas  

Este repositorio contiene mi **primer proyecto de análisis de datos**, donde aplico **Python** y **SQL** para explorar, limpiar y analizar un dataset de precios de viviendas.  
El proyecto incluye la implementación de un **modelo de Regresión** para la **predicción de precios de casas**.  

---

  ## Objetivos  

- Realizar **análisis exploratorio de datos (EDA)** para comprender qué variables influyen en el precio de las casas.  
- Implementar **técnicas de regresión** para segmentar viviendas en grupos con características similares.  
- Construir un **modelo predictivo básico** como introducción a modelos de machine learning para predicción de precios.  
- Desarrollar habilidades prácticas con Python y librerías de análisis de datos.  

---

 ## Herramientas y Tecnologías  

- **Lenguajes:** Python
- **Librerías principales:**  
  - pandas, numpy → limpieza y manipulación de datos  
  - matplotlib, seaborn → visualización  
  - scikit-learn → regresión y modelo predictivo  
- **Entorno:** Google Colab / Jupyter Notebook  

---
# ESTADO DEL PROYECTO 
<h4 align="center">
:construction: Proyecto en construcción :construction:
</h4>

# ESTRUCTURA DEL PROYECTO  

## Portafolio_Housig_PredictionTraining
Se realiza un análisis de datos, para identificaciar las variables más importantes que afectan a la estimación de precios. Lo primero a realizar es el análisis general del *BASE DE DATOS* original, para hacer un análisis exploratorio, se revisa las correlaciones para identificar y elegir las variables  con la mayor correlación, posteriormente se realiza un *DataFrame* con los datos que vamos a utilizar, y a las variables a las que vamos a aplicar el modelo de regresión. 

La base de datos original es proveniente de Kaggle. Se utiliza un modelo de ML basado en modelos de regressión (Decision Tree Regressor, Random Forest Regressor). Este análisis me sirve para mostrar habilidades de análisis de datos, comprensión básica de modelos de automatización de ML. Uso de  Python, SQL, elaboración de Dashboards, y Matplotlib.

## Contexto de los datos
Esta base de datos contienen información del censo de las casas en California de 1990. 
Entonces este trabajo no va aprdecir los datos actuales de las casas, fue utilizado en el segundo capítulo del libro de Aurélien Géron, "Aprendizaje automático práctico con Scikit-Learn y TensorFlow". Se utiliza como ejercicio e introducción a la implementación de algoritmos de aprendizaje automático. Pero la base disponible en Kaggle es un una versión modificada del conjunto de datos de Vivienda de California, disponible en la página de Luís Torgo (Universidad de Oporto).

Objetivo del análisis:
                       * A partir de los datos históricos utilizar técnicas de predicción para obtener un precio apropiado para una casa en función de las variables que inciden en mayor medida en el precio.
                       * Creación de un modelo utilizando datos disponibles para pronosticar los precios de casa.
          
## Entendimiento del negocio
Aunque este análisis es solamente como ejercicio, y demostración de habilidades, el análisis de este Dataset ayuda a ver como un buen análsis de datos, puede generar conocimiento valioso para toma de desiciones en diferentes sectores, ayudando a :

***Objetivo comercial:*** Poder predecir precios de casas de manera automática, basándose en algunas características determinadas. 
Mejorar la toma de decisiones
Reducir de costos
Mejorar la calidad de servicio
Reducir de errores

***Objetivo de análisis y minería de datos:*** A partir de los datos históricos se analizan las variables disponibles para identificar cuáles influyen de manera predominante en el objeto de estudio. Con base en este análisis, se aplican técnicas de predicción (modelo de regresión) que permiten estimar un precio adecuado de una vivienda en función de las variables que mayor impacto tienen en su valor..


## Codigo  Instalaci[on y Ejecución 
Para poder hacer este análisis fue hehco en Google Cola con Pythom, y es necesario intalar las siguientes librerias:

🧰 Requisitos
- Python 3.10+ (recomendado)
- Git
- Google Colab

🚀 Instalación

1) Clona el repositorio
```bash
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO

1. Crea y activa un entorno virtual
Windowspython -m venv .venv
.venv\Scripts\activate

2. Instala dependencias
pip install -r requirements.txt

3. Istala dependencias
pip install -r requirements.txt

▶️ Ejecución
A) Script de Python
python src/main.py --config configs/base.yaml

B) Jupyter Notebook (EDA / análisis)
jupyter lab   # o: jupyter notebook
# Abrir: notebooks/EDA.ipynb

2) Librerías ultilizadas 
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

⏹️ Salir del entorno
deactivate


               

                


