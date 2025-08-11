# Importar os pacotes necessarios
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# Construir o analisador de argumento e analisar os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# Definir os limites inferiores e superiores de cada cor



# Define faixa de cor (ex: vermelho)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Se um caminho de video nao foi fornecido, pegue a referencia webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(2)

# Caso contrario, pegue uma referencia para o arquivo de video
else:
	camera = cv2.VideoCapture(args["video"])

# Manter looping
while True:
	# Agarrar o quadro atual
	(grabbed, frame) = camera.read()

	# Se estamos a ver um video e nos nao pegar um quadro,
	# Em seguida, chegamos ao final do video
	if args.get("video") and not grabbed:
		break

	# Redimensionar o quadro, esbater-lo e converte-lo para o HSV
	# espaco colorido
	frame = imutils.resize(frame, width=600)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Construir uma mascara para a cor verde e outra pra a cor amarela
	# Uma serie de dilatacoes e erosoes para remover qualquer ruido
	maskRed = cv2.inRange(hsv, lower_red, upper_red)
	maskRed = cv2.erode(maskRed, None, iterations=2)
	maskRed = cv2.dilate(maskRed, None, iterations=2)


	# Encontrar contornos da mascara e inicializar a corrente
	cntRed = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	centerRed = None


	# Unica proceder se pelo menos um contorno foi encontrado
	if len(cntRed) > 0:
		# Encontrar o maior contorno da mascara, em seguida, usar-lo para calcular o circulo de fecho minima e
		# centroid
		cRed = max(cntRed, key=cv2.contourArea)
		rectRed = cv2.minAreaRect(cRed)
		boxRed= cv2.boxPoints(rectRed)
		boxRed = boxRed.astype(np.intp)
		mRed = cv2.moments(cRed)
		centerRed = (int(mRed["m10"] / mRed["m00"]), int(mRed["m01"] / mRed["m00"]))
		cv2.drawContours(frame, [boxRed], 0, (0, 0, 255), 2)


	# Mostrar o quadro na tela
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Condicao de parada 'q', parar o loop
	if key == ord("q"):
		break

# Limpeza da camara e feche todas as janelas abertas
camera.release()
cv2.destroyAllWindows()
