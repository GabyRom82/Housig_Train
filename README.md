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
:<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/b25df3b6-ee43-4817-9c9b-67480bff8a15" />
 Proyecto terminado 
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


## 📂 Código, Instalación y Ejecución  

Este análisis fue realizado en **Google Colab** con **Python**, por lo que es necesario contar con las siguientes librerías y herramientas instaladas.

### 🧰 Requisitos
- Python 3.10+ (recomendado)  
- Git  
- Google Colab (o Jupyter Notebook si trabajas localmente)  

---

### 🚀 Instalación

1. **Clona el repositorio**
```git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO

2. **Crea y activa un entorno virtual**
Windows
python -m venv .venv
.venv\Scripts\activate

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

3. **Instala las dependencias**
pip install -r requirements.txt

▶️ Ejecución
A) Script de Python
python src/main.py --config configs/base.yaml

B) Jupyter Notebook / Google Colab
jupyter lab
jupyter notebook

1. Abrir el archivo:
notebooks/EDA.ipynb

📚 Librerías utilizadas
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
```

## Desarolladora del proyecto

**GABRIELA ROMERO  MORENO** 
<img align="right" width="150" alt="Screenshot 2025-07-16 152132" src="https://github.com/user-attachments/assets/735c69ec-20b7-45d4-8646-1497d2e6e7f8" />


MAESTRÍA EN CIENCIAS  
ANALISTA DE DATOS CON MÁS DE 5 AÑOS DE EXPERIENCIA EN ENTORNOS CIENTÍFICOS
EDUCATIVOS-ADMINISTRATIVOS. ESPECIALIZADA EN ANÁLISIS ESTADÍSTICO, GESTIÓN Y 
VISUALIZACIÓN DE BASES DE DATOS.  



## Preparación de los datos

  Revisamos nuestra base de datos para cerciorarnos que no hay valores nulos, ni valores repetidos en nuestra base de datos, y que esta quede limpia para poder aplicar correctamente el modelo.
  
  Debido a que la base de datos original presenta un alto grado de valores faltantes en la variable PooQC usada 99.520548, se reemplazan los valores faltantes por 0 y se agrega la columna que utilizamos en nuestra base ya sin nulos y limpia llamada PoolArea.

Imagen de la base original: <img width="920" height="596" alt="image" src="https://github.com/user-attachments/assets/457e177a-642d-44e0-bf73-752318263b96" />

Imagen de la base Clean_Data lista sin nulos

<img width="515" height="502" alt="image" src="https://github.com/user-attachments/assets/af967a70-9783-42b5-88b4-88333d3e078b" />
<img width="748" height="702" alt="image" src="https://github.com/user-attachments/assets/191b1497-cf52-45f5-83c5-5bcaef966e4b" />


Debido a que las variables que elegimos, todas soon numéricas, no hay necsidad de trasformar variables categóricas a numéricas. 

  


## Análisis Descriptivo

En la base de datos original, se tenía un total de 81 variables , y 1460 datos. Una vez revisado los índices de correlación, se determinó aplicar el modelo solo a 8 variables con el índice más alto de correlación. 

La base de datos limpia, con las 8 variablesse ve de la siguiente forma: 
<img width="987" height="569" alt="image" src="https://github.com/user-attachments/assets/f0d2a882-c4ac-4257-8c06-6d5522f8e30a" />

posee las siguientes carácterísticas: 

<img width="1116" height="453" alt="image" src="https://github.com/user-attachments/assets/ddf33cd6-a370-43d9-9df9-9f2ecf08ca23" />

<img width="830" height="649" alt="image" src="https://github.com/user-attachments/assets/50ea76e7-8a06-44d2-8148-3abedc9ad3b5" />

Y las correlaciones entre nuestra variable objetivo ('SalesPrice') y las dem[as variables> 

<img width="1364" height="422" alt="image" src="https://github.com/user-attachments/assets/d203087c-4bf6-41f0-b128-1b17a3e24672" />
<img width="956" height="434" alt="image" src="https://github.com/user-attachments/assets/da43afb4-2902-420c-96df-e9c2663b04c0" />



## MODELADO (aprendizaje supervisado, Regresión)

Se ***definen valores de X y Y*** para modelo de aprendizaje supervisado:
``` # Definir valores de X y Y para modelo de aprendizaje supervisado
X= df.drop(columns=['SalePrice']).values
y= df['SalePrice'].values
print(f'{X.shape=}')
print(f'{y.shape=}')
```

Al ser un modelo de **aprendizaje supervisado**, es decir, utiliza datos etiquetados para predecir el resultado, debemos *separar el conjunto de datos* en entrenamiento y prueba, con 70/30 para regresión.

<img width="877" height="260" alt="image" src="https://github.com/user-attachments/assets/b79b85e1-e0e7-4814-a983-8b33ed7c5047" />

 
 Para poder comparar dos prediccones distintas, utilizaremos dos modelos diferentes de aprendizaje supervisado: árbol de decisión para regresión y random forest. Mientras que un Árbol de Decisión divide los datos para hacer predicciones de regresión (valor numérico) o clasificación (categoría), el modelo Random Forest mejora la precisión y reduce el sobreajuste combinando las predicciones de múltiples árboles de decisión, obteniendo el resultado final mediante el promedio (regresión) o la votación mayoritaria (clasificación).

¿Por qué es útil usar ambos en el proyecto?

1.Comparación de desempeño: Árbol de decisión → modelo base, sencillo de entender. Random Forest → modelo más robusto, ideal para ver si mejora la precisión.
2.Balance entre interpretabilidad y rendimiento: Con el árbol puedes mostrar las reglas de decisión (útil para explicar a no técnicos). Con Random Forest tienes un modelo más confiable para predicciones reales.
3.Importancia de variables: Ambos permiten medir qué variables afectan más al precio (ej: superficie, ubicación, número de habitaciones). Esto te ayuda a justificar el análisis además de la predicción.

<img width="1271" height="482" alt="image" src="https://github.com/user-attachments/assets/0723b196-f8a2-49ab-a344-75d8743b788b" />

Entonces ahora  se van a **definir** los modelos

``` # Definir modelo: árbol de decisión para regresión
mod_dt_reg= DecisionTreeRegressor(max_depth=2,random_state=0)
#Entrenar
mod_dt_reg.fit(x_train, y_train)
#Predecir
y_pred_dt_reg= mod_dt_reg.predict(x_test)

#Definir el modelo Random Forest para regresión
mod_rf_reg= RandomForestRegressor(max_depth=2, random_state=0)

#Entrenar
mod_rf_reg.fit(x_train, y_train)

# Predecir
y_pred_rf_reg = mod_rf_reg.predict(x_test)
```


Una vez definidos los modelos, podemos pasar  a ***EVALUACIÓN Y ENTRENAMIENTO*** de los modelos.

Para explicar esta parte, primero debemos entender que cuando entrenamos un modelo supervisado (en este caso de regresión), necesitamos medir qué tan bien predice sobre datos que no ha visto. En nuestro caso cuando la variable a predecir es numérica, como el precio de casas, la regresion mide qué tan lejos están las predicciones de los valores reales. 

Métricas comunes:
MAE (Mean Absolute Error / Error Absoluto Medio)
MSE (Mean Squared Error / Error Cuadrático Medio)
RMSE (Root Mean Squared Error / Raíz del Error Cuadrático Medio)

En el ejemplo que presento aquí, el último método RMSE es el que defino como método de evaluaicón, y se determina con la siguiente formula:   <img width="403" height="88" alt="image" src="https://github.com/user-attachments/assets/cea53246-08d9-4e85-9bbc-8f997c5c9db0" />

Ya que mide el error en las mismas unidades de la variable. Es la métrica más usada en árboles de decisión y random forest porque:
*Da una idea clara de cuán lejos, en promedio, está la predicción del valor real.
*Penaliza más los errores grandes (por ejemplo: predecir una casa en 200,000 cuando vale 500,000).

🔸 En específico el RMSE en el Árbol de Decisión para regresiónes es útil porque:
1. Penaliza fuertemente errores grandes (importante en predicciones de precios), es decir,  durante el entrenamiento, el árbol de decisión ***se ajusta minimizando*** el RMSE en cada división.
2. Está en la misma escala de la variable (dinero, metros, etc.), lo que facilita la interpretación.
3. Al evaluar el modelo, usamos RMSE en datos de prueba para ver qué tan bien generaliza.

Con esto quiero decir, en este ejercicio que si predecimos precios de casas y se obtiene un RMSE = 25,000, significa que en promedio, tu modelo se equivoca en ±25,000 dólares respecto al precio real.

El codigo para poder evaluar los modelos:
```#Evaluar modelo de árbol de decisión para regresión: Método RMSE
rmse_dt_reg= np.sqrt(mean_squared_error(y_test, y_pred_dt_reg))
print(f'RMSE Árbol de decisión: {rmse_dt_reg: .2f}')

# Evaluar modelo de random forest para regresión
rmse_rf_reg = np.sqrt(mean_squared_error(y_test, y_pred_rf_reg))
print(f'{rmse_rf_reg = :.2f}')
```
Estás evaluaicones nos indican que para el modelo del Árbol de decisión su RMSE:  48115.47, y para el modelo Random forest su RMSE = 45236.62, recordemos que el RMSE devuelve el error en las mismas unidades de la variable objetivo (en este caso dólares), la interpretación indica que cuanto más bajo sea el RMSE, mejor es el modelo (menor error).

Además tenemos la evaluación de las demás metricas, que nos indican , de igual manera que el modelo de Random forest es mejor:
<img width="479" height="651" alt="image" src="https://github.com/user-attachments/assets/683db7af-2f88-4b09-9093-760d6698f3d0" />



📊 Comparación de Modelos de Regresión
Al ver los valores de evaluación de los dos modelos supervisados para la predicción de precios de vivienda: Decision Tree Regressor y Random Forest Regressor. Los resultados muestran que el Random Forest ofrece un mejor desempeño estadístico frente al Decision Tree.
Este modelo presenta errores más altos y solo explica el 66.8% de la variabilidad, por lo que es menos preciso y generaliza peor que el Random Forest.

✅ Conclusión

El Random Forest Regressor es el modelo que más se acerca a los valores reales, con menor error absoluto y cuadrático y una mayor capacidad explicativa (R²). Esto se debe a que, al combinar múltiples árboles de decisión, reduce la varianza y logra una predicción más robusta frente a datos nuevos.Una vez que evaluamos que modelo es mejor, vamos a guardar los modelos para poder hacer las pruebas de predicción y comprobar que realmente el modelo que se ajusta mejor, es el modelo Random forest.

Una vez evaluados ambos modelos, los guardamos, con el código:
```# Guardar modelo
modelo_rf= mod_rf_reg 
model_type= 'regresion'
library_version= 'rf_sklearn' + '_' + sklearn.__version__.replace('.','_')
model_name= model_type + '_' + library_version + '.pickle'

pickle.dump(modelo_rf, open("regresion_RandomForestRegressor.pkl", "wb"))

print(type(modelo_rf))
print('Saved model: ' + model_name)

# Guardar modelo
modelo_dt = mod_dt_reg
model_type = 'regression'
library_version = 'Decicion_TreeRegressor' + '_' + sklearn.__version__.replace('.','_')
model_name = model_type + '_' + library_version + '.pickle'

pickle.dump(modelo_dt, open("regresion_DecisionTreeRegressor.pkl", "wb"))

print(type(modelo_dt))
print('Saved model: ' + model_name)
```
Para poder proceder a las predicciones, con datos nuevos, y con los datos reales.

***PREDICCIÓN**

Para poder hacer las predicciones, primero creamos una matriz de Numpy, para utilizar el modelo con datos nuevos:

```# Nuevos datos de [	OverallQual,	GrLivArea,	GarageArea,	TotalBsmtSF,	FullBath,	YearBuilt,	PoolArea	]
X_new = np.array([[  9,	1607,	192,	156,	1,	2015,	0]])
x_new1= np.array([[  7,	1710,	548,	856,	2,	2003,	0]])
x_new2= np.array([[6,	1262,	460,	1262,	2,	1976,	0]])
x_new3= np.array([[7,	1786,	608,	920,	2,	2001,	0]])
x_new4= np.array([[5,	1256,	276,	1256,	1,	1965,	0]])
```
En este caso x_new1 y x_new3 corresponden a datos reales, para poder comparar y conocer realmente la precisión de la predicción en ambos modelos. Las predicciones las hacemos con el siguiente códigopoemos verlas: 
 
<img width="706" height="477" alt="image" src="https://github.com/user-attachments/assets/bc9c913c-ae82-4f81-b077-029fb086dbc4" />

<img width="625" height="591" alt="image" src="https://github.com/user-attachments/assets/b8dc2c2e-64f0-4240-bca6-20b56f501591" />

























