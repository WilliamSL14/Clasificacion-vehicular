import cv2
import numpy as np

#Objeto de video
video = cv2.VideoCapture('Relaxing highway traffic.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

#Ciclo infinito
while True:
	#Leer siguiente cuadro
	ret, frame = video.read()
	#Si hay siguiente cuadro, ret es TRUE de lo contrario es false y se rompe el ciclo
	if ret:
		porcentaje_escala = 100 # percent of original size
		width = int(frame.shape[1] * porcentaje_escala / 100)
		height = int(frame.shape[0] * porcentaje_escala / 100)
		dim = (width, height) 
		frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
		#Mostramos el cuadro leído
		cv2.imshow('Video Original', frame)
		#Convertimos el cuadro a escala de grises para procesarlo
		frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('Escala de Grises',frame_gris)		
		#Aplicamos MOG acá
		fgmask = fgbg.apply(frame_gris)		
		#Le hacemos thresholding para eliminar las sombras que el MOG2 deja en gris
		ret,fgmask = cv2.threshold(fgmask,250,255,cv2.THRESH_BINARY)
		#Aplicamos un pequeño filtro para quitarle ruido a la máscara
		fgmask = cv2.GaussianBlur(fgmask,(3,3),0)
		#Ahora vamos a dilatar y erosionar un poco la máscara para poder darle un espacio blanco al objeto
		#no que detecte la forma sino sólo su espacio
		kernel = np.ones((8,8),np.uint8)
		fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
		fgmask = cv2.erode(fgmask,kernel,iterations = 1)
		cv2.imshow('Mascara de MOG2',fgmask)				
		#Ahora hacemos el AND bit a bit de la máscara y la imagen original
		res = cv2.bitwise_and(frame,frame,mask = fgmask)
		cv2.imshow('resultado',res)
		#Hallamos los componentes que comparten bits blancos en la imagen
		contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		#Se dibuja cada contorno si su alto y ancho es mayor a 20 pixeles
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			if((w>20) and (h>20)):
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				ROI = frame[y:y+h, x:x+w]				
		#Mostramos la imagen final
		cv2.imshow('resultado final',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break;
	else:
		break;