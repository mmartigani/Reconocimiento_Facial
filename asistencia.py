#PROYECTO LO QUE VA HACER ES VER A PERSONAS , 
#EN UNA CAMARA WEB Y LAS VA A COMPARAR, 
#CON UNA BASE DE DATOS DE FOTOS QUE TENEMOS GUARDADAS 

#PARA REGISTRAR EL INGRESO O PROHIBICION DE INGRESAR 
import cv2
import face_recognition as fr 
import os
import numpy
from datetime import datetime
 #crear base de datos 
ruta='empleados'
mis_imagenes=[]
nombres_empleados=[]
lista_empleados=os.listdir(ruta)
#lista_empleados=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg',]
#print(lista_empleados)
for nombre in lista_empleados:
    imagen_actual=cv2.imread(f'{ruta}/{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

#codificar imagenes 

def codificar(imagenes):
    #creamos una lista nueva 
    lista_codificada=[]
    #pasar las imgenes a rgb 
    for imagen in imagenes:
        imagen=cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        #codificar
        codificado=fr.face_encodings(imagen)[0]
        #agregar a la lista 
        lista_codificada.append(codificado)
        #devolver lista codificada 
    return lista_codificada
lista_empleados_codificadas=codificar(mis_imagenes)
#print(len(lista_empleados_codificadas))
#REGISTRAR LOS INGRESOS
def registrar_ingresos(persona):
    f=open('registro.csv', 'r+')
    lista_datos=f.readline()
    nombres_registro=[]
    for linea in lista_datos:
        ingreso=linea.split(',')
        nombres_registro.append(ingreso[0])
        
    if persona not in nombres_registro:
        ahora=datetime.now()
        string_ahora=ahora.strftime('%H:%M:%S') 
        f.writelines(f'\n{persona}, {string_ahora}')  
        
        
        
        
#tomar una imagen de camara web 
captura=cv2.VideoCapture(0, cv2.CAP_DSHOW)
#LLEMOS LA IMAGEN DE LA CAMARA 
exito, imagen =captura.read()
if not exito:
    print('no se ha podido tomar la capura')
else:
    #reconocer la cara en la captura 
    cara_captura=fr.face_locations(imagen)
     #codificar la cara capturada 
    cara_captura_codificada=fr.face_encodings(imagen, cara_captura)
    #buscar coincidencias
    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
         coincidencias=fr.compare_faces(lista_empleados_codificadas, caracodif)
         distancias=fr.face_distance(lista_empleados_codificadas, caracodif)
         
         print(distancias)
         indice_coincidencias=numpy.argmin(distancias)
         #mostrar coincidencias 
         if distancias [indice_coincidencias] >0.6:
             print('no coincide con los empleados')
         else:
             #buscar el nombre del empleado 
             nombre=nombres_empleados[indice_coincidencias]
             #cuatro puntos para ubicar la cara 
             y1,x2,y2, x1=caraubic
             cv2.rectangle(imagen,(x1, y1), (x2, y2), (0,255,0),2)
             cv2.rectangle(imagen,(x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
             cv2.putText('imagen web', nombre,(x1 + 6, y2 -6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
             registrar_ingresos(nombre)
             #mostrar la imagen obtenida 
             cv2.imshow('imagen web', imagen)
             
             #mantener ventana abierta 
             cv2.waitKey(0)