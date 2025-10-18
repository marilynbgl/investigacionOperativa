import numpy as np
import libreria as lb 

print('{:15s}{:25s}{:20s}'.format(" n","cond","error Relativo", "error con Gauss"))
print("-"*50)
solutions=[]
for i in range(4,17):
    x=np.ones(i)
    H=lb.hilbert_matrix(i)
    b=H.dot(x)

    c=np.linalg.cond(H,2)# aqui se define las condiciones para la matriz H el 2 indica que se usa la norma euclideana
    xx=np.linalg.solve(H,b)
    err=np.linalg.norm(x-xx, np.inf)/np.linalg.norm(x,np.inf)
    solutions.append(xx)

    print('{:2d}{:20e}{:20e}'.format(i,c,err))