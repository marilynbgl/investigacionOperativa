import numpy as np
import libreria as lb 
import time

print('{:15s}{:25s}{:20s}'.format(" n","cond","error Relativo", "error con Gauss"))
print("-"*50)
solutions=[]
for i in range(4,17):
    x=np.ones(i)
    H=lb.hilbert_matrix(i)
    b=H.dot(x)

    c=np.linalg.cond(H,2)# aqui se define las condiciones para la matriz H el 2 indica que se usa la norma euclideana
    #xx=np.linalg.solve(H,b)
    b=b.reshape(b.shape[0],1)# redimensionan
    xx= lb.GaussElimSimple(H,b)
    err=np.linalg.norm(x-xx, np.inf)/np.linalg.norm(x,np.inf)
    solutions.append(xx)

    #print('{:2d}{:20e}{:20e}'.format(i,c,err))

n=1000
A=np.random.uniform(0,10, size=(n, n))
b=np.random.rand(n, 1)

start_time1=time.perf_counter() #para ver cuanto tiempo se toma en hacer los calculos
X= lb.GaussElimSimple(A,b)
end_time1= time.perf_counter()
elapsed_time1=end_time1-start_time1
normErr1=np.linalg.norm(A@X-b,1)


start_time2=time.perf_counter() #para ver cuanto tiempo se toma en hacer los calculos
X= lb.GaussElimWithPiv(A,b)
end_time2= time.perf_counter()
elapsed_time2=end_time2-start_time2
normErr2=np.linalg.norm(A@X-b,1)

start_time3=time.perf_counter() #para ver cuanto tiempo se toma en hacer los calculos
X= np.linalg.solve(A,b)
end_time3= time.perf_counter()
elapsed_time3=end_time3-start_time3
normErr3=np.linalg.norm(A@X-b,1)


print('{:25s}{:25s}{:25s}'.format("Error Gauss simple","Error Gauss piv","Error linalg.solve"))
print("-"*70)
print('{:20e}{:20e}{:20e}'.format(normErr1,normErr2,normErr3))
# aqui para mostrar los tiempos de calculos comparando los tres metodos
print('{:25s}{:25s}{:25s}'.format("tiempo Gauss simple","tiempo Gauss piv","tiempo linalg.solve"))
print("-"*70)
print('{:20e}{:20e}{:20e}'.format(elapsed_time1,elapsed_time2,elapsed_time3))
#print(f'Tiempo transcurrido: {elapsed_time1:.4f}, {elapsed_time2:.4f}, {elapsed_time3:.4f}  segundos')
