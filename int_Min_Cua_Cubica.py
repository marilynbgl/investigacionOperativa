import numpy as np 
import scipy.linalg as la 
import matplotlib.pylab as plt 
import libreria as lib 

# define true model parameters (ahora con término cúbico)
x = np.linspace(-1.1,2.1, 100)
a, b, c, d = 1, 2, 150, -50     # valores reales
y_exact = a + b*x + c*x**2 + d*x**3

# simulate noisy data 
m = 1000
X = 2 - 3*np.random.rand(m)
Y = a + b*X + c*X**2 + d*X**3 + 4*np.random.randn(m)

# Aquí se construye la matriz A para un polinomio de grado 3
# A = [1, x, x^2, x^3]
At = np.array([X**0, X**1, X**2, X**3])

# Matriz normal At * A
auxMat = np.matmul(At, At.T)

# Vector b = At * Y
b = np.matmul(At, Y).reshape(-1, 1)

# Solución usando tu función de eliminación con pivoteo
sol = lib.GaussElimWithPiv(auxMat, b)

# Extraigo coeficientes
a_fit, b_fit, c_fit, d_fit = sol.flatten()

# Evaluación del polinomio ajustado
y_fit = a_fit + b_fit*x + c_fit*x**2 + d_fit*x**3

# Gráfica
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'r', lw=2, label='True polynomial')
ax.plot(x, y_fit, 'b', lw=2, label='Cubic least squares fit')

ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.legend(loc=2)
plt.show()
