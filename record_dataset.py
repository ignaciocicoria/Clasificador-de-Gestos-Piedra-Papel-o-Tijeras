import cv2
import mediapipe as mp
import numpy as np
import os

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Abrir cámara
cap = cv2.VideoCapture(0)

# Crear listas para guardar datos
data = []
labels = []

# Diccionario de gestos
gestos = {'r': 0, 'p': 1, 's': 2}  # r=piedra, p=papel, s=tijeras

# Crear carpeta para guardar dataset
if not os.path.exists('dataset'):
    os.makedirs('dataset')

print("Presiona 'r'=piedra, 'p'=papel, 's'=tijeras, 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    #espeja la imagen
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    #si detecta alguna mano, dibuja los puntos y conexiones sobre la mano en el frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Grabación Gestos", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key in [ord('s'), ord('p'), ord('r')]:
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.append(lm.x)
                coords.append(lm.y)
            
            # Guardar en listas
            data.append(coords)
            labels.append(gestos[chr(key)])
            print(f"Gesto '{chr(key)}' guardado. Total: {len(labels)}")

# Guardar en archivos .npy
np.save('dataset/rps_data.npy', np.array(data))
np.save('dataset/rps_labels.npy', np.array(labels))

print("Dataset guardado en carpeta 'dataset'.")
cap.release()
cv2.destroyAllWindows()