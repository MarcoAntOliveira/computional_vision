import cv2

# Acessa a webcam (0 = webcam padrão)

cap2 = cv2.VideoCapture(4)



if not cap2.isOpened():
    print("Não foi possível acessar a câmera 1 ")
    exit()


while True:
    # Captura frame por frame

    ret2, frame2 = cap2.read()
  
    if not ret2:
        print("Falha ao capturar o frame")
        break
  
    # Exibe o frame
    cv2.imshow('Webcam2', frame2)
    
    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap2.release()
cv2.destroyAllWindows()
