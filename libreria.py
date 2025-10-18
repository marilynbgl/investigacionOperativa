import numpy as np
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

def sustRegresiva(A,b): #sustitucion regresiva
   N= b.shape[0] #A matriz cuadrada b un arreglo 
   x= np.zeros((N,1))
   for i in range (N-1,-1,-1):
      x[i,0]=(b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
   return x #arreglo bidimensional

def GaussElimSimple(A,b): # si hay ceros en la diagonal no funciona bien es sin pivoteo 
   Ab=np.append(A,b,axis=1) # axis=1 indica agregar una columna al final si es axis=0 agrega una fila al final 
   escalonarMatriz(Ab)
   A1=Ab[:,0:Ab.shape[1]-1].copy()
   b1=Ab[:,Ab.shape[1]-1].copy()
   b1=b1.reshape(b.shape[0],1) # aqui lo convierte en una matriz de ua sola columna
   x=sustRegresiva(A1,b1)
   return x # arreglo bidimensional

def hilbert_matrix(n):# ejemplo de una matriz mal condicionada
  A= np.zeros((n,n))
  for i in range (1,n+1):
     for j in range (1,n+1):
        A[i-1,j-1]=1/(i+j-1)
  return A