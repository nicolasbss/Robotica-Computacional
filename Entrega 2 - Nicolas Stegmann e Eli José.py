
# -*- coding:utf-8 -*-

#LINK PARA VIDEO DO CODIGO FUNCIONANDO NO YOUTUBE:
#  https://youtu.be/jymfFPgeiqQ


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# Para usar o vídeo
#cap = cv2.VideoCapture('hall_box_battery_mp2.mp4')

# As 3 próximas linhas são para usar a webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
kernel = np.ones((30,30),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

def identifica_cor(frame):
    '''
    Segmenta o maior objeto cuja cor é parecida com cor_h (HUE da cor, no espaço HSV).
    '''

    # No OpenCV, o canal H vai de 0 até 179, logo cores similares ao 
    # vermelho puro (H=0) estão entre H=-8 e H=8. 
    # Veja se este intervalo de cores está bom
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cor_menor = np.array([-8, 50, 50])
    cor_maior = np.array([8, 255, 255])
    segmentado_cor = cv2.inRange(frame_hsv, cor_menor, cor_maior)
    segmentado_cor = cv2.morphologyEx(segmentado_cor, cv2.MORPH_OPEN, kernel)

    # Será possível limpar a imagem segmentado_cor? 
    # Pesquise: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html


    # Encontramos os contornos na máscara e selecionamos o de maior área
    img_out, contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    maior_contorno = None
    maior_contorno_area = 0


    cv2.drawContours(frame, contornos, -1, [255, 0, 255], 5)


    for cnt in contornos:
        area = cv2.contourArea(cnt)


        if area > maior_contorno_area:
            maior_contorno = cnt
            maior_contorno_area = area

    
    # Encontramos o centro do contorno fazendo a média de todos seus pontos.
    if not maior_contorno is None :
        cv2.drawContours(frame, [maior_contorno], -1, [0, 0, 255], 5)
        maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
        ymax = 0
        ymin = 100000
        for p in maior_contorno:
     	    if p[1] > ymax:
      	        ymax = p[1]
            if p[1] < ymin:
                ymin = p[1]   
        h = ymax - ymin    
        if h == 0:
        	h = 0.1
        media = maior_contorno.mean(axis=0)
        media = media.astype(np.int32)
        cv2.circle(frame, tuple(media), 5, [0, 255, 0])
        distancia = int(37 * 641 / h)
    else:
        media = (0, 0)
        distancia = 0

    print(distancia)
    cv2.putText(frame,str(distancia),(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('', frame)
    cv2.imshow('imagem in_range', segmentado_cor)
    cv2.waitKey(1)

    centro = (frame.shape[0]//2, frame.shape[1]//2)

    return media, centro, distancia


while(True):
    # Capture frame-by-frame
    print("Novo frame")

    ret, frame = cap.read()


    img = frame.copy()

    media, centron, distancia = identifica_cor(img)
    #print(distancia)
    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('original',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("No circles were found")
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
