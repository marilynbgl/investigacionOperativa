import numpy as np
#A=np.array([[1,2],[3,4]])
#print(A)

A=np.random.randint(0,10, size=(4, 2)) # matriz de orden 6x5
print( A)
#print("\n", A[[2,4],:]) # los : indica toda la fila  extrae toda la fila 2 y 4

def iterCambiaFilas(A,fil_i,fil_j):
    A[[fil_i,fil_j],:]=A[[fil_j,fil_i],:]

def operacionFila(A,fil_m, fil_piv, factor): #fil_m fila modificada fil_piv fila pivotaeda
    A[fil_m,:]= A[fil_m,:]- factor*A[fil_piv,:]

def reescalaFila(A,fil_m,factor): # la fila multiplicada por una constante
     A[fil_m,:]= factor*A[fil_m,:]


#operacionFila(A,2,0,2)
reescalaFila(A,0,3)
print(A)
