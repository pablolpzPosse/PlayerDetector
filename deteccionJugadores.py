# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread("E:/Universidad/187.jpg")

kernel = np.ones((5,5))/25
dst = cv2.filter2D(img,-1,kernel)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
dst1 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(dst,cv2.COLOR_RGB2HSV)

#segmentación imagen

lower_green = np.array([40,40, 40])
upper_green = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
kernel = np.ones((15,15))
op = cv2.erode(mask, kernel)
op = cv2.dilate(op, kernel)
op = cv2.medianBlur(op,11,op)
res = cv2.bitwise_and(dst, dst, mask=op)

res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2RGB)
res_gray = cv2.cvtColor(res_bgr,cv2.COLOR_RGB2GRAY)

#Lineas Campo

hist = cv2.equalizeHist(res_gray)
edges = cv2.Canny(hist,150,200) 
lines = cv2.HoughLinesP(edges, 2, np.pi/180, 110, np.array([]), 100, 20)
for l in lines:
    for x1,y1,x2,y2 in l:
        cv2.line(dst,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.line(dst1,(x1,y1),(x2,y2),(255,255,0),2)
        
#Marcar Jugadores
        
_,thresh = cv2.threshold(res_gray,0,255,cv2.THRESH_BINARY_INV)
clo = cv2.dilate(thresh, kernel)
clo = cv2.erode(clo, kernel)
contours, hierarchy = cv2.findContours(clo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if(w>20 and h>= 1) and (w < 150): 
        cv2.rectangle(dst,(x,y),(x+w,y+h),(0,0,255),3)

    
