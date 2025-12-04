import numpy as np
import libreria as lb 
import time

def medir_tiempo(func, A, b):
    start = time.perf_counter()
    func(A.copy(), b.copy())   # usamos copia para no modificar el original
    end = time.perf_counter()
    return end - start

def medir_tiempoS(func, A):
    start = time.perf_counter()
    func(A.copy())   # usamos copia para no modificar el original
    end = time.perf_counter()
    return end - start

n=500 #tamaño del sistema lineal
#A1=np.random.uniform(0,5, size=(n, n)) # matriz de orden 4x4 con numeros flotantes de 0 a 5
A=np.random.rand(n,n)
b= np.random.uniform(0,6, size=(n,1))
xx= np.array([[2,1,-1,2],[4,5,-3,6],[-2,5,-2,6],[4,11,-4,8]])
y= np.array([[1,-2,-1,3],[-1,3,-2,-2],[2,0,1,1],[1,-2,2,3]])
z= np.array([[1,4,0,0],[2,3,0,1],[0,4,1,5],[0,0,2,3]])
m= np.array([[0,4,-2,4],[-6,2,10,0],[5,8,-5,2],[0,-2,1,0]])
ra= np.array([[1,2,3,-1],[0,5,1,7],[2,-1,5,-9],[4,-2,10,-18]])
#re= np.zeros((n,n))
descpmqr= np.array([[1,1,2],[1,0,-2],[-1,2,3]])
descomp2= np.array([[2,1,0,0],[1,2,1,0],[0,1,2,1],[0,0,1,2]])
invers=np.array([[2,-1,3],[4,0,1],[6,1,2]])


d=lb.Determinante(A)
d2=np.linalg.det(A)
print('{:28s}{:25s}'.format("determinate método","determinate con numpy"))
print("-"*50)
print('{:20e}{:25e}'.format(d,d2))

r=lb.Rank(ra)
r1= np.linalg.matrix_rank(ra)
print('{:28s}{:25s}'.format("rango método","rango con numpy"))
print("-"*50)
print('{:20e}{:25e}'.format(r,r1))

r2=lb.Inversa(invers)
print("la inversa es:\n",r2)
r3= np.linalg.inv(invers)
print("la inversa con numpy es:\n",r3)

QR= lb.qr_factorization(A)
Q=QR[0]
print("Q: \n", Q)
R=QR[1]
print("R: \n",R)
print("A=QR: \n", Q@R)

QR2=np.linalg.qr(A)
Q2=QR2[0]
print("Q: \n", Q2)
R2=QR2[1]
print("R: \n",R2)
print("A=QR: con numpy \n", Q2@R2)
#print("la matriz Q2 es:\n",Q2)

A2=np.array([[1,0,0,1],[0,1,-1,-1],[0,-1,1,0],[1,-1,0,0]])
b2=np.array([[0],[0],[1],[0]])
X1= lb.solveByQR(A2,b2)
print(X1)

A3=np.array([[4,-1,0,0,0],[-1,4,-1,0,0],[0,-1,4,-1,0],[0,0,-1,4,-1],[0,0,0,-1,4]])
b3=np.array([[100],[200],[200],[200],[100]])
X2= lb.solveByQR(A3,b3)
print(X2)

# 1. mi método
t1 = medir_tiempoS(lb.qr_factorization,A)
# 2. Numpy
t2 = medir_tiempoS(np.linalg.qr, A)
#print(f"Tiempo qr factorizacion: {t1:.6f} s")
#print(f"Tiempo numpy.linalg.qr: {t2:.6f} s")
print('{:25s}{:25s}'.format("Tiempo qr factorizacion","Tiempo numpy.linalg.qr"))
print("-"*50)
print('{:20e}{:25e}'.format(t1,t2))


t3 = medir_tiempo(lb.GaussElimWithPiv,A,b)
t4 = medir_tiempo(lb.solveByLU,A,b)
t5 = medir_tiempo(lb.solveByQR,A,b)
t6 = medir_tiempo(np.linalg.solve,A,b)
# aqui para mostrar los tiempos de calculos comparando los tres metodos
print('{:28s}{:20s}{:20s}{:20s}'.format("time Gauss piv","time LU","time QR","tiempo linalg.solve"))
print("-"*70)
print('{:15.6f}{:20.6f}{:20.6f}{:20.6f}'.format(t3,t4,t5,t6))

x=lb.GaussElimWithPiv(A2,b2)
normErr1=np.linalg.norm(A2@x-b2,1)
x2=lb.solveByQR(A2,b2)
normErr2=np.linalg.norm(A2@x2-b2,1)
x3=np.linalg.solve(A2,b2)
normErr3=np.linalg.norm(A2@x3-b2,1)
# aqui para mostrar los tiempos de calculos comparando los tres metodos
print('{:28s}{:20s}{:20s}'.format("Error Gauss piv","Error QR","Error linalg.solve"))
print("-"*70)
print('{:20e}{:20e}{:20e}'.format(normErr1,normErr2,normErr3))

