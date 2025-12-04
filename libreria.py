import numpy as np
def interCambiaFilas(A,fil_i,fil_j):
    A[[fil_i,fil_j],:]=A[[fil_j,fil_i],:]

def operacionFila(A,fil_m, fil_piv, factor): #fil_m fila modificada fil_piv fila pivotaeda
    A[fil_m,:]= A[fil_m,:]- factor*A[fil_piv,:]

def reescalaFila(A,fil_m,factor): # la fila multiplicada por una constante
     A[fil_m,:]= factor*A[fil_m,:]

def escalonarMatriz(A): #aqui no funciona si en la diagonal principal hay ceros
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

def sustRegresiva(A,b): #sustitucion regresiva RESUELVE UN SISTEMA ESCALONADA CON A TRIANGULAR SUPERIOR
   N= b.shape[0] #A matriz cuadrada b un arreglo 
   x= np.zeros((N,1))
   for i in range (N-1,-1,-1):
      x[i,0]=(b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
   return x #arreglo bidimensional

def sustProgresiva(A,b): #sustitucion progresiva RESUELVE UN SISTEMA ESCALONADA CON A TRIANGULAR inferior
   N= b.shape[0] #A y b matriz cuadrada b un arreglo 
   x= np.zeros((N,1))
   for i in range (0,N):
      x[i,0]=(b[i,0]-np.dot(A[i,0:i],x[0:i,0]))/A[i,i]
   return x #arreglo bidimensional

def GaussElimSimple(A,b): # si hay ceros en la diagonal no funciona bien es sin pivoteo 
   Ab=np.append(A,b,axis=1) # axis=1 indica agregar una columna al final si es axis=0 agrega una fila al final 
   escalonarMatriz(Ab)
   A1=Ab[:,0:Ab.shape[1]-1].copy()
   b1=Ab[:,Ab.shape[1]-1].copy()
   b1=b1.reshape(b.shape[0],1) # aqui lo convierte en una matriz de ua sola columna
   x=sustRegresiva(A1,b1)
   return x # arreglo bidimensional

def GaussElimWithPiv(A,b):
   Ab= np.append(A,b,axis=1) #matriz aumentada b se pega como una columna al final de A, si es axis=0 agrega una fila al final 
   escalonaConPiv(Ab)
   A1= Ab[:,0:Ab.shape[1]-1].copy()
   b1= Ab[:,Ab.shape[1]-1].copy()
   b1= b1.reshape(b.shape[0],1) # fuerza a b1 a tener la forma (n,1), es decir, un vector columna.
   x= sustRegresiva(A1,b1)
   return x # array bidimensional  (n,1) 

def hilbert_matrix(n):# ejemplo de una matriz mal condicionada
  A= np.zeros((n,n))
  for i in range (1,n+1):
     for j in range (1,n+1):
        A[i-1,j-1]=1/(i+j-1)
  return A

def LUdeComposition(A): # para matrices cuadradas 
   nrows= A.shape[0]
   U=A.copy()
   L= np.eye(nrows,nrows,dtype=np.float64) # matriz de numeros decimales y es la matriz identidad solo en la diagonal hay unos
   for col in range (0,nrows-1): # AQUI VA A CORRER HASTA LA PENULTIMA COLUMNA
      for row in range (col+1,nrows):
         mult=U[row,col]/U[col,col]
         L[row,col]=mult
         operacionFila(U,row,col,mult)
   return (L,U)

def solveByLU(A,b): # USANDO LA DESCOMPOSICION LU 
   LU= LUdeComposition(A) 
   L=LU[0]
   U=LU[1]
   Y=sustProgresiva(L,b)
   X=sustRegresiva(U,Y)
   return X # arreglo bidimensional

def transpuesta(A):
    A = A.astype(float)  # trabajar con decimales
    filas, columnas = A.shape
    At = np.zeros((columnas, filas))  # crear matriz transpuesta
    for i in range(filas):
        for j in range(columnas):
            At[j, i] = A[i, j]
    return At

def escalonaConPivot(A):
    """
    Escalona una matriz A usando pivoteo parcial seguro.
    Devuelve la cantidad de intercambios de filas.
    """
    nfila, ncol = A.shape
    nCambios = 0
    tol = 1e-12  # tolerancia para evitar divisiones por cero

    for j in range(min(nfila, ncol)):
        # Encuentra el índice del pivote máximo absoluto desde la fila j hacia abajo
        imax_rel = np.argmax(np.abs(A[j:nfila, j]))
        imax = j + imax_rel

        # Si el pivote máximo es muy pequeño, saltar la columna (dependencia lineal)
        if abs(A[imax, j]) < tol:
            continue  # no se puede usar esta columna para eliminar

        # Intercambiar filas si es necesario
        if imax != j:
            interCambiaFilas(A, j, imax)
            nCambios += 1

        # Eliminar debajo del pivote
        for i in range(j + 1, nfila):
            if abs(A[j, j]) > tol:
                ratio = A[i, j] / A[j, j]
                operacionFila(A, i, j, ratio)

    return nCambios

def Rank(A):
   A= A.copy().astype(float)
   escalonaConPivot(A)
   tolerancia= 1e-12
   rango= np.sum(np.abs(np.diag(A))>tolerancia)
   return rango

def multiDeterminante(A):
   d=1
   for k in range(len(A)): 
      d*=A[k][k]
   return d

def Determinante(A):
   #A = A.astype(np.float)
   A = A.astype(np.float64)
   n = A.shape[0]   
   escalonaConPiv(A)          
   return multiDeterminante(A)

def Inversa(A): # metodo de gauss jordan
    n = A.shape[0]
    M = np.hstack((A.astype(float), np.eye(n)))  # matriz aumentada [A | I]
    escalonaConPiv(M)

    # Hacer ceros arriba del pivote (Gauss-Jordan)
    for j in range(n - 1, -1, -1):
        piv = M[j, j]
        if abs(piv) < 1e-12:
            print(" La matriz no tiene inversa.")
            return None
        M[j, :] /= piv  # convierte el pivote en 1
        for i in range(j):
            ratio = M[i, j]
            operacionFila(M,i,j,ratio) #M[i, :] -= ratio * M[j, :]

    A_inv = M[:, n:]
    return A_inv

def producto_interno(u, v):
    return np.dot(u, v)

def gram_schmidt(A):
    A = A.astype(float).copy()
    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        R[j, j] = np.sqrt(np.sum(A[:, j]**2))

        if R[j, j] == 0:
            print("La matriz tiene columnas linealmente dependientes")
            return None, None
        
        Q[:, j] = A[:, j] / R[j, j]

        for k in range(j + 1, n):
            R[j, k] = producto_interno(Q[:, j], A[:, k])
            A[:, k] = A[:, k] - Q[:, j] * R[j, k]

    return Q, R

def qr_factorizationF(A):
    return gram_schmidt(A)

def qr_factorization(A):
    A = A.astype(float)# aseguramos que sea tipo float
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        # r_jj = norma de la j-ésima columna de A
        R[j, j] = np.sqrt(np.sum(A[:, j]**2))
        
        # Si la norma es cero → columnas linealmente dependientes
        if R[j, j] == 0:
            print("A tiene columnas linealmente dependientes")
            return None, None
        
        # Normaliza la columna j de A → genera la columna j de Q
        Q[:, j] = A[:, j] / R[j, j]
        
        # Para cada columna siguiente k > j
        for k in range(j + 1, n):
            # r_jk = producto punto entre Q[:,j] y A[:,k]
            R[j, k] = np.dot(Q[:, j], A[:, k])
            
            # Actualiza A[:,k] eliminando la componente en dirección de Q[:,j]
            A[:, k] = A[:, k] - Q[:, j] * R[j, k]
                 
    return Q, R

def solveByQR(A, b): #Resuelve Ax = b usando descomposición QR
    QR = qr_factorization(A)
    Q=QR[0]
    R=QR[1]
    Qtb = np.dot(transpuesta(Q), b)
    x = sustRegresiva(R, Qtb)
    return x # arreglo bidimensional

