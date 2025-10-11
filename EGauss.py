import numpy as np
#A=np.array([[1,2],[3,4]])
#print(A)

#print("\n", A[[2,4],:]) # los : indica toda la fila  extrae toda la fila 2 y 4

def interCambiaFilas(A,fil_i,fil_j):
    A[[fil_i,fil_j],:]=A[[fil_j,fil_i],:]

def operacionFila(A,fil_m, fil_piv, factor): #fil_m fila modificada fil_piv fila pivotaeda
    A[fil_m,:]= A[fil_m,:]- factor*A[fil_piv,:]

def reescalaFila(A,fil_m,factor): # la fila multiplicada por una constante
     A[fil_m,:]= factor*A[fil_m,:]

def escalonarMatriz(A):
 nfila=A.shape[0] # el shape reconice de un arreglo 
 ncol=A.shape[1]

 for j in range(0,nfila):
     for i in range(j+1, nfila):
         ratio=A[i,j]/A[j,j] # el cociente entre el elemento eliminado con el pivote
         operacionFila(A,i,j,ratio)

def escalonaConPiv(A):
    nfila=A.shape[0] # el shape reconoce de un arreglo las filas y columnas 
    ncol=A.shape[1]

    for j in range(0,nfila):
      imax=np.argmax(np.abs(A[j:nfila,j]))#argmax me da el indice de la fila que tendra el mayor valor absoluto, abs saca el absoluto de toda la columna 
      interCambiaFilas(A,j+imax,j)
      for i in range(j+1, nfila):
         ratio=A[i,j]/A[j,j] # el cociente entre el elemento eliminado con el pivote
         operacionFila(A,i,j,ratio)

def ceros_en_diagonal(A):
  
    nfila=A.shape[0] # el shape reconice de un arreglo 
    ncol=A.shape[1]
    for i in range(nfila):
     for j in range(ncol):
      # Si el índice de la fila es igual al de la columna, es un elemento de la diagonal
      if i == j:
        A[i][j] = 0
    return A

A=np.random.uniform(0,10, size=(4, 5)) # matriz de orden 4x2 con numeros flotantes de 0 a 10
#A[i][i]=0. # aqui en la parte de ratio la division se da entre cero y las respuestas salen nan o inf 
print(A)
#operacionFila(A,2,0,2)
#interCambiaFilas(A,1,2)
#reescalaFila(A,0,3)
#escalonarMatriz(A)
B=ceros_en_diagonal(A)
print("\n",B)
escalonaConPiv(B)
print("\n",B)
