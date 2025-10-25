import numpy as np
import libreria as lb

n=40
A= np.random.rand(n,n)
b= np.random.rand(n,1)
#LU= lb.LUdeComposition(A)
#L=LU[0]
#print("L: \n", L)
#U=LU[1]
#print("U: \n",U)
#print("A-LU: \n", A-L@U)
X= lb.solveByLU(A,b)

print("||Ax-b||_1:\n", np.linalg.norm(A@X-b,1))

