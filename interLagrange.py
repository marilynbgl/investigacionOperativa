import numpy as np
import libreria as lib 
import matplotlib.pyplot as plt
import numpy.polynomial as P


x = np.array([2,3,4,5,7,9])
#y = np.array([1,3,-4,-5,17,-19])
y = np.sin(x)

pol = lib.interpLagrange(x,y)

print(pol)
print(pol(x))

a = x.min()
b = x.max()

xx = np.linspace(a,b,200)
yy = pol(xx)
yy_Exacta = np.sin(xx)

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(xx,yy,'b',lw=2, label='Polinomio interpolante')
ax.plot(x,y,'ro', alpha=0.6, label= 'Datos')
ax.plot(xx,yy_Exacta,'g',lw=2, label='Polinomio solucion Exacta')
ax.legend(loc=2)
#ax.set_xlabel(r"$x$", frontsize=10)
#ax.set_ylabel(r"$y$", frontsize=10)
plt.grid()
plt.show()
