import math
import numpy as np
import random as ran

#class LinAlgebra:

    #def __init__():



def MatrixMult(mat1, mat2):
        prodMat = np.zeros((mat1.shape[1],mat2.shape[0]))
        if mat1.shape[0] == mat2.shape[1] and  mat1.shape[1] == mat2.shape[0]:

            for x in range(mat1.shape[1]):
                
                for i in range(mat1.shape[0]):
                    sum = 0
                    for j in range(mat1.shape[1]):
                        
                        sum +=  mat1[x][j] * mat2[j][i]
                    #print(sum)
                    prodMat[x][i] = int(sum)
                
            print(prodMat)
            return prodMat

  


        else:
            print("matrices do not match")

import numpy as np

mata = np.array([[2,3,4], [3,5,2], [2,3,4]])
matb = np.array([[2,8,4], [3,5,9], [7,3,4]])    

testM = MatrixMult(mata, matb)

#print(testM)

print(np.matmul(mata,matb))

def makeMatrix(dim1, dim2):
    
    matrix = np.zeros((dim1, dim2))
    for i in range(dim1):
        for j in range(dim2):
            matrix[i][j] = ran.randint(1,10)
    return matrix


mat1 = makeMatrix(7, 7)

mat2 = makeMatrix(7, 7)

print( MatrixMult(mat1, mat2))


print( np.matmul(mat1, mat2))

def matrixAdd()


