import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo entrenado
MODEL_PATH = "rps_model.h5"
model = load_model(MODEL_PATH)

# Diccionario de clases
CLASSES = {0: "Piedra", 1: "Papel", 2: "Tijeras"}

# Umbral de confianza mínimo
CONF_THRESHOLD = 0.5

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Espejar la imagen
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            prediction = "Esperando mano ('q' para salir)"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks de la mano
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extraer coordenadas (x, y) de los 21 landmarks
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y])  # 42 valores

                    # Convertir a numpy array
                    input_data = np.array(coords).reshape(1, -1)

                    # Predicción
                    probs = model.predict(input_data, verbose=0)
                    max_prob = np.max(probs)
                    class_id = np.argmax(probs)

                    # Verificar umbral de confianza
                    if max_prob >= CONF_THRESHOLD:
                        prediction = CLASSES[class_id]
                    else:
                        prediction = "Gesto no claro"

            # Mostrar resultado en pantalla (letra negra)
            cv2.putText(frame, f"Gesto: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("Rock-Paper-Scissors", frame)

            # Salir con tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
