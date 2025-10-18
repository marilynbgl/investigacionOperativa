import numpy as np
import libreria as lb 
import time

#A=np.array([[1,2],[3,4]])
#print(A)

#print("\n", A[[2,4],:]) # los : indica toda la fila  extrae toda la fila 2 y 4

n=800  #tamaño del sistema lineal
A=np.random.uniform(0,6, size=(n, n)) # matriz de orden 4x2 con numeros flotantes de 0 a 10
b= np.random.uniform(0,6, size=(n,1))
#A[i][i]=0. # aqui en la parte de ratio la division se da entre cero y las respuestas salen nan o inf 
#print(A)
#operacionFila(A,2,0,2)
#interCambiaFilas(A,1,2)
#reescalaFila(A,0,3)
#escalonarMatriz(A)
#B=ceros_en_diagonal(A)
#print("\n",B)
#escalonaConPiv(B)
#print("\n",B)
"""
A=np.array([[1,2,1],[1,0,1],[0,1,2]])
b=np.array([[0],[2],[1]])
print(A)
print(b)"""
start_time=time.perf_counter() #para ver cuanto tiempo se toma en hacer los calculos
sol= lb.GaussElimSimple(A,b)
end_time= time.perf_counter()
elapsed_time=end_time-start_time
print(f'Tiempo transcurrido: {elapsed_time:.4f} segundos')
#print("\n",sol)
residuo= A@sol - b
#print("\n",residuo)
print("norma del residuo",np.linalg.norm(residuo))


