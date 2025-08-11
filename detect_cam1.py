import cv2
import numpy as np

cap = cv2.VideoCapture(2)
if not cap.isOpened():

    
    print("Não foi possível acessar a câmera 1 ")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define faixa de cor (ex: vermelho)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Máscara para a cor
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Aplica máscara
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Detecção de Cor Vermelha", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
