# -*- coding: utf-8 -*-
"""
Código para elimar fondo de un video de tráfico 
vehicular y mostrar sólo los vehiculo en del mismo.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

##### Carga del video:
captura = cv2.VideoCapture('Relaxing highway traffic.mp4')

##### Aplicación de métodos para eliminar el fondo del video:

"""Método MOG:"""
fgbg  = cv2.bgsegm.createBackgroundSubtractorMOG()

""""Método MOG2:"""
#fgbg = cv2.createBackgroundSubtractorMOG2()

"""Método GMG:"""
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#fgbg = cv2.createBackgroundSubtractorGMG()

while ( 1 ): 
    
##### Visualización de la mascara aplicada al video
    ret,frame  =  captura.read()
    mascara =  fgbg.apply(frame)
    #mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)# Para GMG

##### Configurar el tamaño de ventan:    
    escala = 40
    ancho = int(frame.shape[1]*escala/100)
    alto = int(frame.shape[0]*escala/100)
    dimensiones = (ancho,alto)
    frame1 = cv2.resize(frame,dimensiones, interpolation = cv2.INTER_AREA)

    if ret == True:
        cv2.imshow('video', frame1)
        if cv2.waitKey(30) == ord('d'):
            break
    else: break

##### Aplicacción del filtrado:
    
##Desenfoque Gaussiano:
    #blur = cv2.GaussianBlur(frame,(5,5),0)
    
## Filtro bilateral:    
    #blur = cv2.bilateralFilter(mascara,9,75,75)
    
##Median filtering:
    #median = cv2.medianBlur(frame,5)
  
## Promedio:    
    blur = cv2.blur(frame,(5,5))
    
##### Visualización de la señal filtrada:   
    plt.subplot(211),plt.imshow(mascara),plt.title('Original')
    plt.xticks([]),plt.yticks([])
    plt.subplot(212),plt.imshow(blur),plt.title('Filtro')
    #plt.subplot(212),plt.imshow(median),plt.title('Filtro')# Para median.
    plt.xticks([]), plt.yticks([])
    plt.show()
    
##### Obtención de los vehículos sin fondo:
    imgMask = cv2.bitwise_and(frame,frame,mask = mascara)
    imgMask = cv2.resize(imgMask,dimensiones, interpolation = cv2.INTER_AREA)
    cv2.imshow('Selección',imgMask)
    
captura.release() 
cv2.destroyAllWindows() 