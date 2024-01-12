import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer un archivo Excel en un DataFrame de pandas
archivo_excel = 'contaminacion.xlsx'
df = pd.read_excel(archivo_excel)

x = df['Year'].values  # Años
y = df['Annual CO2‚ emissions (Colombia)'].values

# Inicializar listas para almacenar valores conocidos y desconocidos
anios_conocidos = []
valores_conocidos = []
anios_desconocidos = []

# Separar los valores conocidos y desconocidos
for k in range(len(x)):
    if np.isnan(y[k]):  # Si el valor es NaN (desconocido)
        anios_desconocidos.append(x[k])  # Almacenar el año en desconocidos
    else:
        anios_conocidos.append(x[k])  # Almacenar el año en conocidos
        valores_conocidos.append(y[k])  # Almacenar la emisión en conocidos

# Función para calcular los coeficientes de los splines cúbicos
def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x) #Arreglo que almacena la diferencia que hay entre los elementos de x
    df = np.diff(y) / h #se saca la derivada

    # Matriz tridiagonal
    A = np.zeros((n-1, n-1)) #se crea matriz de 0
    #no se incluye el primero y el ultimo debido a que hace parte de los puntos externos y tienen un efecto indirecto en la suavidad
    #de la funcion. Mientras que los puntos internos si tienen un impacto directo
    np.fill_diagonal(A, 2*(h[:-1] + h[1:]))
    np.fill_diagonal(A[1:], h[1:-1])
    np.fill_diagonal(A[:, 1:], h[1:-1])

    # Lado derecho (o sea el arreglo que quedaba de (3/h1)*(a2-a1) de la clase)
    B = 3 * (df[1:] - df[:-1])

    # Resolver sistema tridiagonal
    C = np.zeros(n+1)
    C[1:-1] = np.linalg.solve(A, B)
    # Coeficientes de los splines cúbicos. Se usa la formula de la profe
    a = y[:-1]
    b = df - h * (2*C[:-1] + C[1:])/3
    c = C[:-1]
    d = (C[1:] - C[:-1]) / (3 * h)

    # Retorna los coeficientes en forma de una lista de tuplas
    return [(a[i], b[i], c[i], d[i]) for i in range(n)]



# Función para evaluar los splines cúbicos
def eval_spline(splines, x_values, t):
    n = len(splines)
    for i in range(n):
        if x_values[i] <= t <= x_values[i + 1]:
            a, b, c, d = splines[i]
            return a + b * (t - x_values[i]) + c * (t - x_values[i]) ** 2 + d * (t - x_values[i]) ** 3

# Calcular los splines cúbicos para los valores conocidos
splines = cubic_spline(anios_conocidos, valores_conocidos)

# Calcular los valores interpolados para los años desconocidos
valores_interpolados = [eval_spline(splines, anios_conocidos, t) for t in anios_desconocidos]

valores_interpolados = [eval_spline(splines, anios_conocidos, t) for t in anios_desconocidos]

for i, t in enumerate(anios_desconocidos):
    interpolacion = valores_interpolados[i]
    print(f"Interpolación para el año {t}: {interpolacion}")


# Crear una gráfica
plt.figure(figsize=(10, 6))
plt.plot(anios_conocidos, valores_conocidos, 'bo-', label='Datos conocidos')
plt.plot(anios_desconocidos, valores_interpolados, 'ro', label='Datos interpolados')

plt.xlabel('Año')
plt.ylabel('Emisión CO2')
plt.title('Interpolación con Splines Cúbicos de las Emisiones de CO2')
plt.legend()
plt.grid(True)
plt.show()