import cv2


for i in range(5):  # testando de 0 a 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Câmera disponível no índice {i}")
        cap.release()
    else:
        print(f"Nenhuma câmera no índice {i}")
