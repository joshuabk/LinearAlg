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


mat1 = makeMatrix(5, 5)

mat2 = makeMatrix(5, 5)

print( MatrixMult(mat1, mat2))


print( np.matmul(mat1, mat2))

def matrixAdd(mat1, mat2):
    sumMat = np.zeros((mat1.shape[1],mat2.shape[0]))
    if mat1.shape[1] == mat2.shape[1] and  mat1.shape[0] == mat2.shape[0]:
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                sum = mat1[i][j] +  mat2[i][j]
                
                sumMat[i][j]  = sum

        return sumMat


def matrixSub(mat1, mat2):
    sumMat = np.zeros((mat1.shape[1],mat2.shape[0]))
    if mat1.shape[1] == mat2.shape[1] and  mat1.shape[0] == mat2.shape[0]:
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                diff = mat1[i][j] -  mat2[i][j]
                
                sumMat[i][j]  = diff 

        return sumMat


sumM = matrixAdd(mat1, mat2)

def getAdjoint(mat):
    adjMat = np.zeros((mat.shape[1],mat.shape[0]))
    for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                adjMat[j][i] = mat[i][j]
    return adjMat

def getCofactorSubmatrix(mat, row, col):
    subMat = []
    for i in range(mat.shape[0]):
            for j in mat.shape[0]:
                if i != row and j != col:
                    row.append(mat[i][j])

            subMat.append(row)
    return subMat

def sumCofactMat(mat):
    for i in range(mat.shape[0]+1):
        for j in range(mat.shape[0]+1):
            if i == mat.shape[0]:
                i = 0

            
        





def getCofactor(mat):
    if mat.shape[0] ==mat.shape[1]:

        cofMat = np.zeros((mat.shape[0], mat.shape[0]))
        for i in mat.shape[0]:
            for j in mat.shape[0]:

    else:
        print("matrix is not square")


print(sumM)
print(np.add(mat1, mat2))


invM = getAdjoint(mat1)

print(mat1)

print(invM)

print(np.transpose(mat1))

class subMatrix:
    def __init__(self, coef, mat):
        self.coef = coef
        self.mat = mat


    def determinate(self):
        subMatList = []
        size = self.mat.shape[0]
        mat = self.mat
        if size > 2:
            for x in range(size):
                Ncoef = mat[0][x]
                NewMat = []
                if x%2 ==0:
                    sign = 1
                else:
                    sign = -1
                for i in range(1,size):
                    row = []
                    for j in range(size):
                        if  j != x:
                            row.append(mat[i][j])

                    NewMat.append(row)

                print(NewMat)
                newSub = subMatrix(sign* self.coef*Ncoef, np.array(NewMat))
                subMatList.append(newSub)
            print(subMatList)
            return subMatList
 

testSub = subMatrix(1, mat1)
newList = testSub.determinate()

def getListOfTwos(matList):
    while(matList[0].mat.shape[0]>2):
        newList = []
        for mat in matList:
            for item in mat.determinate():
                newList.append(item)

        matList = newList
    return matList
twoList = getListOfTwos(newList)

print(twoList)
i = 0
for mat in twoList:
    print(mat.mat)
    print(mat.coef) 
    i += 1

def calcDeterminantFromTwos(matList):
    total= 0
    for mat in matList:
        total += mat.coef*(mat.mat[0][0] * mat.mat[1][1] - mat.mat[0][1] * mat.mat[1][0])

    return total


determin =  calcDeterminantFromTwos(twoList)

print(determin)
np
print(np.linalg.det(mat1))

print(invM/determin)
matAdj = np.matrix(mat1)
print(np.linalg.inv(mat1))
print(matAdj.getH())
print(invM)