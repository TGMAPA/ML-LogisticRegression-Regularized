'''
Miguel Ángel Pérez Ávila 

Archivo con implementación de las funciones solicitadas
'''

# Importar Librerias y modulos
from MLDev import MLDev
import numpy as np

# ========== Funciones ==========

# Función para graficar la disperción de datos (Dados los Vectores X y Y) y la función ajustada
# Está función ASUME que el vector de X aún contiene la columna X0 del bias con "unos"  
# - Input:
#       - x: Vector de X 
#       - y: Vector de Y 
#       - theta: Vector de thetas óptimas p
# - Return : Void
def graficaDatos(x, y, theta):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    mlObject.scatterPlotAndRegression6Degree(x,y,theta) # Ejecución del método implementado para la graficación

# Función para optimizar y conseguir el vector de thetas optimo que ajuste a los vecotres de datos dados a razon
# de un paso (alpha) definido y una n cantidad de iteraciones
# - Input:
#       - theta: Vector de thetas iniciales para el entrenamiento
#       - x: Vector de X 
#       - y: Vector de Y 
#       - iteraciones: epocas a ejecutar
#       - aplha: Learning Rate o paso para el ajuste de pesos
#       - lmbda: constante de regularización
# - Return : 
#       - thetas: np.array de thetas óptimas
def aprende(theta, X, y, iteraciones, alpha = 0.01, lmbda = 1):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML

    # Calculo de tetas optimas después del entrenamiento
    thetas = mlObject.gradDescRegLog(iteraciones, theta, X, y, alpha, lmbda=lmbda, regulate=True)

    print("\nMostrando gráfico de Error a través de ",iteraciones," epocas...")
    mlObject.graph(np.arange(iteraciones), mlObject.Costos, "Epochs", "Error: J(theta)", "J(theta) vs Epocas")
    
    return thetas

# Función para realizar predicciones utilizando las thetas optimas del modelo
# y la matriz de X 
# - Input:
#       - theta: Vector de thetas a evaluar
#       - x: Vector de X 
#       - y: Vector de Y 
#       - lmbda: constante de regularización
# - Return : 
#       - Y_predicted
def predice(theta, X, umbral = 0.5):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML

    # Realizar predicciones como el calculo vectorizado de y_hipotesis = X*theta
    p = np.dot(X, theta)

    # Aplicar función sigmoide para los valores resultantes y así normalizar
    p = mlObject.sigmoid(p)

    # Aplicar umbral
    Y_predicted = (p >= umbral).astype(int)

    return Y_predicted
    
# Función para calcular el costo dado un Vector de X, Y y thetas y la gradiente correspondiente
# - Input:
#       - theta: Vector de thetas a evaluar
#       - x: Vector de X 
#       - y: Vector de Y 
#       - lmbda: constante de regularización
# - Return : 
#       - Costo
#       - Gradiente
def funcionCostoReg(theta, X, y , lmbda):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML

    # Calculo del costo con regularizacion
    J = mlObject.calcCostRegLog(theta, X, y, lmbda, regulate=True)

    # Calculo vectorizacdo de la gradiente 
    grad = mlObject.calcGradRegLog(theta, X, y)
    
    return J, grad

# Función para calcular el valor de la funcion sigmoide para un valor z
# - Input:
#       - z
# - Return : 
#       - 1/(1+np.exp(-z))
def sigmoidal(z):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    return mlObject.sigmoid(z) # Resultado de la funcion sigmoide aplicada a z

# Método para extender una matriz con 2 predictores (X) a grado 6
# La matriz de predictores X no se pasa con la columna de bias integrada, todo se construye en esta función
# - Input:
#       - x: Vector de X 
# - Return : 
#       - mappedmatrix
def mapeoCaracteristicas(X, tipo="Manual"):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML

    # Definir manualmente vector de X sin columna de bias
    mlObject.setXVector(X)

    # ELegir el método deseado para la construcción de la matriz al grado 6
    if tipo == "Manual":
        mappedmatrix = mlObject.manualbuildPolynomialX(mlObject.X)
    elif tipo == "scikit-learn":
        mappedmatrix = mlObject.buildPolynomialX(6, mlObject.X, True)

    return mappedmatrix

# ========== Fin de Funciones ==========
