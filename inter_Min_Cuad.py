import numpy as np 
import scipy.linalg as la 
import matplotlib.pylab as plt 
import libreria as lib 

# define true model parametrers 
xa, xb = -3,5 # intervalo
x = np.linspace(xa,xb,100)
a, b, c, d = 1,2,5,4
y_exac = a + b*x + c*x**2 + d*x**3

# simulate noisy data 
m =4000
X = xa + (xb-xa)*np.random.rand(m)
Y = a + b*X + c*X**2 + d*X**3 + 20*np.random.randn(m)

# fit the data to the model using linear least square
#A = np.vstack([X**0, X**1, X**2]) 
#sol, r, rank, sv = la.lstsq(A.T, Y)


At = np.array([X**0, X**1, X**2, X**3])
auxMat = np.matmul(At,At.T)
np.reshape(Y,(m,1))
b = np.matmul(At, Y)
b = b.reshape(-1,1)
sol = lib.GaussElimWithPiv(auxMat,b)


y_fit = sol[0] + sol[1] *x + sol[2]*x**2 +  sol[3]*x**3
fig, ax, = plt.subplots(figsize=(12,4))

ax.plot(X,Y, 'go', alpha=0.5, label='Simuled date')
ax.plot(x,y_exac,'r', lw=2, label='True value $y =1 +2x + 5x^2+ 4x^3$')
ax.plot(x,y_fit,'b', lw=2, label='Least square fit')
#ax.set_xlabel(r,"$x$", fontsize=18)
#ax.set_ylabel(r,"$y$", fontsize=18)
ax.legend(loc=2)
plt.show()