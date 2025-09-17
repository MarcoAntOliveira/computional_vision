import cv2
import numpy as np
from keras.models import load_model

# Carregar modelo treinado
model = load_model("color_model.h5")

# Labels das classes (ordem igual à pasta do treinamento)
class_labels = ['classe1', 'classe2', 'classe3', 'classe4', 'classe5',
                'classe6', 'classe7', 'classe8', 'classe9']

# Abrir câmera (0 = webcam padrão)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processamento (igual ao treino)
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predição
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    # Mostrar no frame
    label = f"{class_labels[class_id]}: {confidence*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento em tempo real", frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
