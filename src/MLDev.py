'''
Miguel Ángel Pérez Ávila

Estructura para el control de procesos Machine Learning
'''

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import chardet
from sklearn.preprocessing import PolynomialFeatures

# Estructura para ejecutar procesos de Machine Learning 
class MLDev:
    
    # Metodo constructor de la clase para declarar los atributos de las instancias y recibir el path del archivo de datos
    def __init__(self):
        self.X = None
        self.Y = None
        self.Costos = None

    # Definir vector preConstruido para X - Considerando la existencia de columna bias x0 con unos.
    def setXVector(self, X):
        self.X = X

    # Definir veector preConstruido  para Y
    def setYVector(self, Y):
        self.Y = Y
    

    # ---- Lectura e interpretación de archivos para la extracción de datos
    # Método para obtener el encoding del archivo de texto
    def getFileEncoding(self, path):
        # Detectar codificación
        with open(path, "rb") as f:
            raw_data = f.read(10000)
        encoding = chardet.detect(raw_data)["encoding"]

        return encoding

    # Método para realizar la lectura del archivo y llenar los arreglos de X y Y
    # - Se agrega automaticamente la columna de X0 con valores "unos" al vector X
    def readFile(self, path, chr2split, omitColumnTitles = True, addX0Col = True):
        # Abrir archivo para lectura
        file = open(path, "r", encoding=self.getFileEncoding(path)) 
        self.X = []
        self.Y = []

        if omitColumnTitles:
            firstLine = True
        else:
            firstLine = False

        for line in file.readlines():
            if not firstLine:
                # Leer linea por linea y llenar la matriz de X y el arreglo de Y evitando el encabezado del dataset y asumiendo
                # que la columna de Y es la ultima en el dataset y Agregando la columna de 0 para X0 o bias

                line = line.split(chr2split)

                aux_xArray = [] # Arreglo o temporal correspondiente a cada muestra del dataset (renglon)
                
                if addX0Col:
                    aux_xArray.append(1.0)

                for i in range(len(line)):
                    if i == len(line)-1: # Ultimo elemento = Valor de y
                        self.Y.append( float(line[i]) )
                    else:
                        # Cualquier otro elemento = Valor de Xi
                        aux_xArray.append( float(line[i]))

                # Agregar arreglo temporal a la matriz X en construcción
                self.X.append(aux_xArray) 
                
            else:
                firstLine = False

        # Cast de list() a np.array
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Y = self.Y.reshape(self.Y.shape[0], 1) # Redimensionar para corregir (m, ) a (m, 1)

    # Metodo para dividir el dataset en conjunto de prueba y entrenamiento para X y Y
    def dataset_division(self, train_percentage):

        X_train = self.X[:int(train_percentage*len(self.X))]
        X_test = self.X[int(train_percentage*len(self.X)):]

        Y_train = self.Y[:int(train_percentage*len(self.Y))]
        Y_test = self.Y[int(train_percentage*len(self.Y)):]

        return X_train, X_test, Y_train, Y_test
    # ---- END Lectura e interpretación de archivos para la extracción de datos


    # ---- Perceptron Simple
    # Método para recalcular Pesos a partir de una Xi y un error=!=0 acierto paso definido
    def recalcW(self, W, X_train, Xi, step, error):
        for i in range(len(W)):
            W[i] = W[i] + (step * error * X_train[Xi][i])

    # Método para entrenar y obtener los pesos ideales para un dataset 
    def simplePerceptron(self, step, train_percentage, n_epochs, f, W = np.array([]), RandW = True, graph = False):
        # Dividir dataset en arreglos de entrenamiento y de prueba
        X_train, X_test, Y_train, Y_test = self.dataset_division(train_percentage)

        if(RandW): # Calcular pesos aleatoriamente
            # Crear arreglo de pesos de dimensiones (N_VariablesXi, 1)
            W = np.random.uniform(low=0, high=1, size=(X_train.shape[1], 1))
        
        print("W Inicial:  \n", W)

        errors = np.array([]) # Arreglo para acumular errores por epoca
        epochs = np.arange(n_epochs) # Arreglo secuencial de 0-n_epochs

        # Iteración por epocas
        for i in range(n_epochs):
            # Aplicar multiplicación de matrices en vez de la sumatoria
            WX = np.dot(X_train, W)

            n_sample = 0 # Contador del número de muestra actual (numero de renglon Xi)
            for wx in WX: # Aplicar funcion de activación y calcular el error para cada muestra
                error = Y_train[n_sample]-f(wx)
                
                if  error == 0: # Error=0
                    pass
                else:
                    # Error != 0
                    # Recalcular Pesos
                    self.recalcW( W, X_train, n_sample, step, error)
                    break
                n_sample+=1

            # Acumular error obtenido en esta epoca, ya sea error!=0 o error==0
            errors = np.append(errors, error)     


        print("\nW Final: \n", W)

        W0 = W[0]
        W = W[1:]

        # CCrear funcion vecotrizada
        f = np.vectorize(f)


        # Predicción de datos de prueba
        X_test = np.delete(X_test, 0 ,axis= 1)
        Y_test_predicted = (np.dot(X_test, W)) + W0
        Y_test_predicted = f(Y_test_predicted)
        Y_test_predicted = Y_test_predicted.flatten()

        print("\n Predicción de datos de prueba")
        i = 0
        for y_predicted, y_desired in zip(Y_test_predicted, Y_test):
            print(i, " Desired: ", y_desired, "  | Predicted: ", y_predicted, "  | IsCorrect: ", y_desired==y_predicted)
            i+=1


        # Predicción de datos de entrenamiento para corroborar entrenamiento
        Y_train_predicted =  (np.dot(np.delete(X_train, 0 ,axis= 1), W)) + W0
        Y_train_predicted = f(Y_train_predicted)
        Y_train_predicted = Y_train_predicted.flatten()
        print("\n Predicción de datos de Entrenamiento")
        i = 0
        for y_predicted, y_desired in zip(Y_train_predicted, Y_train):
            print(i, " Desired: ", y_desired, "  | Predicted: ", y_predicted, "  | IsCorrect: ", y_desired==y_predicted)
            i+=1

        # Graficar Error
        if graph:
            self.graph(epochs, errors, "Epochs", "Error", "Error vs epochs")

        print("\nFinish")
    # ---- End Perceptron Simple
    

    # ---- Reg Lineal
    # Método para obtener el vector de Theta/Pesos Optimos mediante Gradiente Descendiente  - Regresión Lineal
    def gradDesc(self, n_epochs, alpha, thetas ):
        # Mostrar dimensiones de thetas y X
        print("\nThetas Shape: ", thetas.shape)
        print("X Shape     : ", self.X.shape)

        # Definir m como el tamaño de muestras que hay
        m = self.X.shape[0]

        # Arreglo de de almacenamiento de costos
        self.Costos = []
        
        # Redimensionar Y de (m, ) a (m,1)
        self.Y = self.Y.reshape(m, 1)
        print("Y Shape     : ", self.Y.shape)
        
        # Iteración por numero de epocas
        for i_epoch in range(n_epochs):
            # Calcular la "y-gorrito" como y_gorrito = X*Theta de forma vectorial
            y_hypothesis = np.dot(self.X, thetas)

            # Calcular el error 
            error = y_hypothesis - self.Y

            # Calculo de la gradiente unicuamente como  gradiente = (1/m)*error*X
            grad = (1/m) * np.dot(np.transpose(self.X), error)
            
            # Calculo final de thetas como  thetas = thetas - alpha * gradiente
            thetas = thetas - alpha*grad

            # Calculo del costo con las thetas actuales
            costo = self.calcCost(self.X, self.Y, thetas)
            self.Costos.append(costo)

        # Cast de list() a np.array
        self.Costos = np.array(self.Costos)
        
        return thetas

    # Función que calcula el costo para un vector de X. Y y thetas dadas - Regresión Lineal
    def calcCost(self, x, y, thetas):
        # (97,2)*(2,1) = (97,1)
        f_hypothesis = (np.dot(x, thetas))  
        
        # ((97, 1) - (97, 1))**2 = (97, 1)
        toSum = ( f_hypothesis - y )**2   

        # Finalmente J(thetas) = (1/2m)(sumatoria)
        costo = ((1/(2*len(y)))*np.sum(toSum)) 

        return costo
    # ---- End - Reg Lineal


    # ---- Graphs
    # Método para graficar dos variables
    def graph(self, x = None, y = None, xlabel="", ylabel="", title= ""):   
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    # Método para graficar 
    def scatterPlotAndLine(self, x, y, thetas):
        # ELiminar colmna del bias
        x = np.delete(x, 0 ,axis= 1) 

        # Dibujar dispersión
        plt.scatter(x, y, color="blue", label="Datos reales") 

        # Vector de valores de X 
        x = np.linspace(min(x), max(x), 100)  

        # Vector de valores resultantes de la ecuación y = theta0 + theta1 * x
        y = thetas[0] + thetas[1] * x    

        # Graficación
        plt.plot(x, y, color="red")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.title("Gráfico: Dispersión de Datos y Función Ajustada")
        plt.show()
    # ---- End Graphs


    # ---- Reg Log
    # Método para extender una matriz con n cantidad de predictores (X) a un grado deseado
    # La matriz de predictores X no se pasa con la columna de bias integrada, más bien se llama
    # al método con el parametro include_bias = True
    def buildPolynomialX(self, degree, X, include_bias):
        # Generar características polinomiales de grado 2
        poly = PolynomialFeatures(degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)

        return X_poly
    
    # Método para extender una matriz con 2 predictores (X) a grado 6
    # La matriz de predictores X no se pasa con la columna de bias integrada
    def manualbuildPolynomialX(self, X):

        xbias = np.ones(shape=(len(X), 1)) # Crear columna bias
        x0 = X[:,:1]
        x1 = X[:,1:2]
        '''
        [
        '1' 
        'x0' 
        'x1' 
        'x0^2' 
        'x0 x1' 
        'x1^2' 
        'x0^3' 
        'x0^2 x1' 
        'x0 x1^2' 
        'x1^3'
        'x0^4' 
        'x0^3 x1' 
        'x0^2 x1^2' 
        'x0 x1^3' 
        'x1^4' 
        'x0^5' 
        'x0^4 x1'
        'x0^3 x1^2' 
        'x0^2 x1^3' 
        'x0 x1^4' 
        'x1^5' 
        'x0^6' 
        'x0^5 x1' 
        'x0^4 x1^2'
        'x0^3 x1^3' 
        'x0^2 x1^4' 
        'x0 x1^5' 
        'x1^6']
        '''
        X_poly_arr = [
            xbias, 
            x0,
            x1,
            x0**2,
            x0 * x1,
            x1**2,
            x0**3,
            x1 * x0**2,
            x0 * x1**2,
            x1**3,
            x0**4,
            x1 * x0**3,
            (x0**2)*(x1**2),
            x0 * x1**3,
            x1**4,
            x0**5,
            x1 * x0**4,
            (x0**3)*(x1**2),
            (x0**2)*(x1**3),
            x0 * (x1**4),
            x1**5,
            x0**6,
            x1 * (x0**5),
            (x0**4)*(x1**2),
            (x0**3)*(x1**3),
            (x0**2)*(x1**4),
            x0 * (x1**5),
            x1**6
        ]

        # Mezclar arreglos en una sola matriz
        matrix = np.hstack(X_poly_arr) 
        
        return matrix
    
    # Método para calcular la función sigmodial
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    # Método para calcular el costo J con posibilidad de agregar regularización mediante una constante lambda
    def calcCostRegLog(self, theta, X, y, lmbda = 1, regulate=True):
        # Y_hipotesis
        hx = self.sigmoid(np.dot(X, theta))

        # Dimsension de muestras
        m = X.shape[0]

        # Calculo de costos vectorizado 
        J = (-1/m) * np.sum((np.dot(y.T, np.log(hx))) + (np.dot(1-y.T, np.log(1-hx))))

        # Control de aplicación de regulación al costo
        if regulate:
            # Aplicar regularización sin considerar theta 0
            regul = (lmbda / (2*m)) * np.sum(theta[1:]**2)  
            
            # Completar calculo de costa agregando la regularización
            J = J + regul

        return J
    
    # Método para el calculo vectorizado de la gradiente dados los vectores X, Y y theta
    def calcGradRegLog(self, theta, X, y):
        # Y_hipotesis
        hx = self.sigmoid(np.dot(X, theta))

        # Dimsension de muestras
        m = X.shape[0]

        # Calcular el error de forma independiente
        error = hx - y

        # Calcular la gradiente vectorizada
        grad = (1/m) * np.dot(X.T, error)

        return grad

    # Método para obtener el vector de Theta/Pesos Optimos mediante Gradiente Descendiente  - Regresión Logistica
    def gradDescRegLog(self,  n_epochs, thetas, X, y, alpha, lmbda = 1, regulate=True):
        # Dimsension de muestras
        m = X.shape[0]

        # Arreglo de Costos por cada epoca
        self.Costos = []

        for epoch in range(n_epochs):
            # Calculo independiente de la gradiente
            grad = self.calcGradRegLog(thetas, X, y)

            if regulate:
                # Calcular regularización en caso de que así se desee excepto para theta 0
                grad[1:] = grad[1:] + (lmbda/m)*thetas[1:]
        
            # Actualizar thetas con gradiente descendente
            thetas = thetas - alpha * grad

            # Guardar el valor del costo por epoca

            J = self.calcCostRegLog(thetas, X, y, lmbda, regulate=regulate)
            self.Costos.append(J)

        return thetas
    
    # Método para calcular y para una funcion polinomial de sexto grado con 2 variables
    def y6Degree2Variables(self, x0, x1, thetas):
        y = thetas[0][0] + thetas[1][0]*x0 + thetas[2][0]*x1 + thetas[3][0]*(x0**2) + thetas[4][0]*x0*x1 + thetas[5][0]*(x1**2) + thetas[6][0]*(x0**3) + thetas[7][0]*x1*(x0**2) + thetas[8][0]*x0*(x1**2) + thetas[9][0]*(x1**3) + thetas[10][0]*(x0**4)
        
        y+= thetas[11][0]*x1*(x0**3) + thetas[12][0]*(x0**2)*(x1**2) + thetas[13][0]*x0*(x1**3) + thetas[14][0]*(x1**4) + thetas[15][0]*(x0**5) + thetas[16][0]*x1*(x0**4) + thetas[17][0]*(x0**3)*(x1**2) + thetas[18][0]*(x0**2)*(x1**3) + thetas[19][0]*x0*(x1**4) + thetas[20][0]*(x1**5)

        y+= thetas[21][0]*(x0**6)+ thetas[22][0]*x1*(x0**5) + thetas[23][0]*(x0**4)*(x1**2) + thetas[24][0]*(x0**3)*(x1**3) + thetas[25][0]*(x0**2)*(x1**4) + thetas[26][0]*x0*(x1**5) + thetas[27][0]*(x1**6)
        
        return y
    
    # Método para graficar una dispersión 
    def scatterPlotAndRegression6Degree(self, x, y, thetas):
        # Eliminar la columna del bias
        x = np.delete(x, 0, axis=1)  

        x1 = x[:,0]
        x2 = x[:,1]

        # Graficar con diferentes marcadores según y
        plt.scatter(x[y.ravel()==1, 0], x[y.ravel()==1, 1], 
                    marker='x', color='red', label='y=1')
        plt.scatter(x[y.ravel()==0, 0], x[y.ravel()==0, 1], 
                    marker='o', color='blue', label='y=0')

        # Dibujar funcion
        u = np.linspace(x1.min(), x1.max(), 100)
        v = np.linspace(x2.min(), x2.max(), 100)
        U, V = np.meshgrid(u, v)

        # Evaluar polinomio en la malla
        Z = self.y6Degree2Variables(U, V, thetas)

        # Dibujar la curva de la frontera (donde y=0)
        plt.contour(U, V, Z, levels=[0], linewidths=2, colors='green')

        # Labels y leyenda
        plt.xlabel("X1-Prueba1")
        plt.ylabel("X2-Prueba2")
        plt.grid(True)
        plt.legend()
        plt.title("Regresión logística con Función (grado 6)")
        plt.show()
    # ---- End Reg Log