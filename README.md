#  Clasificador de Gestos — Piedra, Papel o Tijeras

##  Descripción

Este proyecto implementa un sistema completo de **reconocimiento de gestos** de “piedra”, “papel” o “tijeras” utilizando **MediaPipe** para la detección de manos y una **red neuronal densa** para la clasificación de los gestos.

El sistema se desarrolla en tres etapas principales:

1️⃣   Se capturan las coordenadas de los landmarks de la mano utilizando MediaPipe y se guardan en un archivo junto con su etiqueta correspondiente.

2️⃣ **Entrenamiento del clasificador de gestos**  
   Se entrena una red neuronal densa sobre las coordenadas de los 21 puntos clave de la mano para clasificar los gestos en tres categorías:  
   - `0`: Piedra  
   - `1`: Papel  
   - `2`: Tijeras  

3️⃣ **Prueba del sistema completo**  
   Se integra la detección de landmarks y el modelo entrenado para reconocer gestos en tiempo real a través de la cámara web.

---

##  Objetivo

Desarrollar un sistema de clasificación capaz de reconocer en tiempo real los gestos de **piedra**, **papel** o **tijeras**, combinando **visión por computadora** y **Deep Learning**.

---

##  Tecnologías utilizadas

-  **Python**
-  **TensorFlow / Keras** — red neuronal densa
-  **MediaPipe** — detección de landmarks de la mano
-  **OpenCV** — captura y visualización de video
-  **NumPy** — manejo de arrays y datasets

---

##  Estructura del proyecto

   ```
  ├── dataset/ # Archivos .npy generados al grabar datos (opcional)
  ├── requirements.txt # Dependencias necesarias
  ├── record_dataset.py # Script para grabar dataset
  ├── train-gesture-classifier.py # Entrena y guarda el modelo
  ├── rock-paper-scissor.py # Ejecuta la inferencia en tiempo real
  └── rps_model.h5 # Modelo entrenado (se genera automáticamente)
   ```
## Preparación del entorno

1️⃣ Crear un entorno virtual
Es recomendable aislar las dependencias del proyecto:

python -m venv venv
Activar el entorno:

En Windows:

venv\Scripts\activate
2️⃣ Instalar dependencias
pip install requirements.txt
3️⃣ Verificar instalación
pip list
Esto mostrará todas las librerías instaladas y sus versiones actuales.

## Modo de uso

1️⃣ **Grabación del dataset (opcional)**  
   Si se desea grabar un nuevo dataset, ejecutar:
   ```
   python record-dataset.py
   ```
   Durante la grabación, realizar los gestos de piedra, papel y tijeras, y presionar la letra correspondiente para grabar el gesto.  
   El script utiliza MediaPipe para detectar los landmarks (21 puntos clave de la mano) y almacena las coordenadas junto con sus etiquetas en archivos .  
   En este repositorio ya se incluyen los archivos `rps_dataset.npy` y `rps_labels.npy`, por lo que este paso puede omitirse.

2️⃣ **Entrenamiento del modelo (opcional)**  
   Si se grabo de nuevo el dataset debe volver a ejecutar el modelo, para ello ejecutar:
   ```
   python train-gesture-classifier.py
   ```
   Carga los datos generados (o los ya incluidos) y entrena una red neuronal densa para clasificar los gestos.  
   El modelo resultante se guarda como `rps_model.h5`.  
   Este repositorio ya  dispone del archivo `rps_model.h5`, por lo que este paso también puede omitirse.

3️⃣ **Prueba del sistema completo**  
   Ejecutar:
   ```
   python rock-paper-scissors.py
   ```
   El sistema:
   - Utiliza MediaPipe para detectar la mano mediante la cámara.
   - Extrae los landmarks y los pasa al modelo entrenado.
   - Muestra en pantalla el gesto reconocido: piedra, papel o tijeras, si el gesto no es claro muestra gesto no claro.


