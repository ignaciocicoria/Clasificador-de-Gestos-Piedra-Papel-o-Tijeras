#  Clasificador de Gestos — Piedra, Papel o Tijeras

##  Descripción

Este proyecto implementa un sistema completo de **reconocimiento de gestos** de “piedra”, “papel” o “tijeras” utilizando **MediaPipe** para la detección de manos y una **red neuronal densa** para la clasificación de los gestos.

El sistema se desarrolla en tres etapas principales:

1. **Grabación del dataset de gestos**  
   Se capturan las coordenadas de los landmarks de la mano utilizando MediaPipe y se guardan en un archivo junto con su etiqueta correspondiente.

2. **Entrenamiento del clasificador de gestos**  
   Se entrena una red neuronal densa sobre las coordenadas de los 21 puntos clave de la mano para clasificar los gestos en tres categorías:  
   - `0`: Piedra  
   - `1`: Papel  
   - `2`: Tijeras  

3. **Prueba del sistema completo**  
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

rock-paper-scissors/
│.
├── record-dataset.py            # Grabación del dataset con MediaPipe.
├── train-gesture-classifier.py  # Entrenamiento de la red neuronal.
├── rock-paper-scissors.py       # Ejecución del sistema completo.
│.
├── rps_dataset.npy              # Coordenadas de landmarks (dataset grabado).
├── rps_labels.npy               # Etiquetas de gestos correspondientes (dataset grabado).
├── rps_model.h5                 # Modelo entrenado (generado por train-gesture-classifier.py).
│.
└── images/                      # Imágenes que muestran el funcionamiento del sistema.


## Modo de uso

1. **Grabación del dataset (opcional)**  
   Si se desea grabar un nuevo dataset, ejecutar:
   ```
   python record-dataset.py
   ```
   Durante la grabación, realizar los gestos de piedra, papel y tijeras, y presionar la letra correspondiente para grabar el gesto.  
   El script utiliza MediaPipe para detectar los landmarks (21 puntos clave de la mano) y almacena las coordenadas junto con sus etiquetas en archivos .  
   En este repositorio ya se incluyen los archivos `rps_dataset.npy` y `rps_labels.npy`, por lo que este paso puede omitirse.

2. **Entrenamiento del modelo (opcional)**  
   Si se grabo de nuevo el dataset debe volver a ejecutar el modelo, para ello ejecutar:
   ```
   python train-gesture-classifier.py
   ```
   Carga los datos generados (o los ya incluidos) y entrena una red neuronal densa para clasificar los gestos.  
   El modelo resultante se guarda como `rps_model.h5`.  
   Este repositorio ya  dispone del archivo `rps_model.h5`, por lo que este paso también puede omitirse.

4. **Prueba del sistema completo**  
   Ejecutar:
   ```
   python rock-paper-scissors.py
   ```
   El sistema:
   - Utiliza MediaPipe para detectar la mano mediante la cámara.
   - Extrae los landmarks y los pasa al modelo entrenado.
   - Muestra en pantalla el gesto reconocido: piedra, papel o tijeras, si el gesto no es claro muestra gesto no claro.


