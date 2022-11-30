import math
import numpy as np

#class LinAlgebra:

    #def __init__():



def MatrixMult(mat1, mat2):
        prodMat = np.zeros(int(mat1.shape[1]), int(mat2.shape[0]))
        if mat1.shape[0] == mat2.shape[1] and  mat1.shape[1] == mat2.shape[2]:

            for x in range(mat1.shape[1]):
                for i in range(mat1.shape[0]):
                    sum = 0
                    for j in range(mat1.shape[1]):
                        sum += mat1[j][x] * mat2.shape[i][j]
                    prodMat[x][i] = sum
                
            print(prodMat) 
            return prodMat

  


        else:
            print("matrices do not match")

import numpy as np

mata = np.array([[2,3], [3,5]])
matb = np.array([[1,5], [3,4]])    

testM = MatrixMult(mata, matb)

