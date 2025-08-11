import cv2

# Acessa a webcam (0 = webcam padrão)
cap1 = cv2.VideoCapture(2)


if not cap1.isOpened():
    print("Não foi possível acessar a câmera 1 ")
    exit()



while True:
    # Captura frame por frame
    ret1, frame1 = cap1.read()
    
    if not ret1:
        print("Falha ao capturar o frame")
        break
   
    # Exibe o frame
    cv2.imshow('Webcam1', frame1)
   
    
    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap1.release()
cv2.destroyAllWindows()
