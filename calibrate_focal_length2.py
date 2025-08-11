import cv2
import numpy as np
import imutils

# Parâmetros de calibração
KNOWN_DISTANCE = 30.0  # cm
KNOWN_WIDTH = 9.5      # cm (largura real do objeto)

# Faixa de cor vermelha em HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Acessar webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro: não foi possível acessar a câmera.")
    exit()
cv2.namedWindow("Calibração", cv2.WINDOW_NORMAL)  # ← ADICIONE AQUI

while True:
    ret, frame = camera.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara para o vermelho
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        width_in_frame = rect[1][0]  # largura detectada (em pixels)
        if width_in_frame > 0:
            focal_length = (width_in_frame * KNOWN_DISTANCE) / KNOWN_WIDTH
            cv2.putText(frame, f"Focal: {focal_length:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"Distância focal estimada: {focal_length:.2f}")
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    cv2.imshow("Calibração", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
