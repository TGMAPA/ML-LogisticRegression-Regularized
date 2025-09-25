'''
Miguel Ángel Pérez Ávila 

Archivo main con un ejemplo de uso en la función main()

'''

# Importar Librerias y modulos
from MLDev import MLDev
from RegularizedLogisticRegression import *
import numpy as np


# Función main para la ejecución de ejemplo
def main():

    # ====== Ejemplo de ejecución de las funciones solicitadas ======

    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    path = "../data/ex2data2.txt"

    # Lectura de archivo mediante un caracter separador considerando que no hay encabezados para las columnas y sin agregar la columna de "unos" o bias
    # y posteriormente se realiza el llenado de vectores 
    mlObject.readFile(path, ",", omitColumnTitles=False, addX0Col=False)  

    # Construccion de matriz (118,2) a grado 6 (118,28) utilizando el métod manual de construcción
    mlObject.X = mapeoCaracteristicas(mlObject.X)

    # Vector de thetas iniciales tamaño (n_variables, 1)
    thetas = np.zeros((mlObject.X.shape[1], 1))
    print("\nThetas Iniciales: \n", thetas.T)

    # Constante de regularización
    LMBDA = 1

    # Ejemplo de Calculo de Costo para Thetas iniciales
    J, grad = funcionCostoReg(thetas, mlObject.X, mlObject.Y, lmbda=LMBDA)
    print("\nError para thetas Iniciales: ",J)

    # Thetas óptimas resultantes del Entrenamiento en busca de thetas óptimas mediante gradiente descendente
    thetas = aprende(thetas, mlObject.X, mlObject.Y, 20000, alpha = 0.01, lmbda=LMBDA)
    print("\nThetas Final    : \n", thetas)
    
    # Ejemplo de Predicciones 
    p = predice(thetas, mlObject.X)

    # Contador de predicciones correctas
    correctos = 0   
    for i in range(len(p)):
        if p[i][0] == mlObject.Y[i][0]:
            # Correcto
            correctos+=1

    # Accuracy del modelo
    print("Accuracy: ",correctos/len(p)*100, "%")

    # Graficación de dispersión con función ajustada visible
    graficaDatos(mlObject.X, mlObject.Y, thetas)

# Ejecutar Main
main()