# ML-LogisticRegression-Regularized: Regresión Logística con Regularización

Este proyecto presenta la **implementación manual de un modelo de regresión logística regularizada** en Python, utilizando **gradiente descendente** para optimizar los parámetros del modelo (θ). La implementación incluye un enfoque orientado a objetos con la clase `MLDev`, así como funciones auxiliares para el cálculo de la sigmoide, la construcción polinomial de características, el entrenamiento regularizado y la visualización de resultados.

El trabajo busca ilustrar los efectos de la **regularización (λ)** en el rendimiento del modelo, comparando escenarios de **overfitting (λ=0)**, **balance óptimo (λ=1)** y **underfitting (λ=100)**.

---

## Objetivos
- Implementar un modelo de **regresión logística regularizada** sin depender de librerías externas de ML.  
- Optimizar parámetros mediante **gradiente descendente vectorizado**.  
- Construir un **mapeo polinomial de características a grado 6**.  
- Analizar el impacto de la **regularización λ** en el ajuste del modelo.  
- Visualizar las fronteras de decisión y la evolución del costo a lo largo de las épocas.  

---

## Funcionalidades implementadas

### Clase `MLDev`
- **Carga y preparación de datos**
  - `getFileEncoding(path)`: Detecta la codificación de archivos.
  - `readFile(path, separador)`: Lee datasets y construye matrices X (features) y Y (etiquetas).
  - `manualbuildPolynomialX(X)`: Expande matrices (m,2) a grado 6 (m,28).
  - `buildPolynomialX(degree, X, include_bias)`: Expansión polinomial con `scikit-learn`.

- **Entrenamiento**
  - `gradDescRegLog(n_epochs, thetas, X, Y, alpha, lmbda=1, regulate=True)`: Entrenamiento con regularización.
  - `calcCostRegLog(theta, X, Y, lmbda=1, regulate=True)`: Cálculo de costo regularizado.
  - `calcGradRegLog(theta, X, Y)`: Gradiente vectorizado.

- **Visualización**
  - `graph(x, y, xlabel, ylabel, title)`: Gráfica evolución del error.  
  - `scatterPlotAndRegression6Degree(X, Y, thetas)`: Dispersión y frontera de decisión.  

### Funciones auxiliares
- `sigmoidal(z)`: Cálculo de la función sigmoide.  
- `aprende(theta, X, Y, iteraciones, alpha=0.01, lmbda=1)`: Gradiente descendente regularizado.  
- `predice(theta, X, umbral=0.5)`: Predicciones y cálculo de accuracy.  
- `funcionCostoReg(theta, X, Y, lmbda)`: Costo y gradiente regularizado.  
- `graficaDatos(X, Y, thetas)`: Gráfica de dispersión y frontera ajustada.  
- `mapeoCaracteristicas(X, tipo="Manual")`: Expansión de características hasta grado 6.  

---

## Tecnologías utilizadas
- **Python 3.11.13**  
- **NumPy**: Operaciones matriciales y vectorización.  
- **Matplotlib**: Gráficas de dispersión, curvas y fronteras de decisión.  
- **Chardet**: Detección de codificación de archivos.  
- **Scikit-learn** (opcional): Construcción de polinomios de características.  

---
 ## Ejecución
1. Clonar este repositorio:  
   ```bash
   git clone https://github.com/TGMAPA/ML-LogisticRegression-Regularized.git
   cd ML-LogisticRegression-Regularized/src
   python main.py
