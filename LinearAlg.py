import math
import numpy as np
import random as ran
import time

#class LinAlgebra:

    #def __init__():




def time_dec(func):
    def getTime(*args, **kwargs):
        start_time = time.time()  
        result  = func(*args,  **kwargs)
        
        timer = time.time() - start_time
        if timer > .10:

            print(func.__name__)
            print(timer)
        
        return result
    return getTime


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
                
            #print(prodMat)
            return prodMat

        else:
            print("matrices do not match")

import numpy as np

mata = np.array([[2,3,4], [3,5,2], [2,3,4]])
matb = np.array([[2,8,4], [3,5,9], [7,3,4]])    

testM = MatrixMult(mata, matb)

#print(testM)

#print(np.matmul(mata,matb))

def makeMatrix(dim1, dim2):
    
    matrix = np.zeros((dim1, dim2))
    for i in range(dim1):
        for j in range(dim2):
            matrix[i][j] = ran.randint(1,10)
    return matrix


mat1 = makeMatrix(9, 9)

mat2 = makeMatrix(2, 2)

#print( MatrixMult(mat1, mat2))
#print( np.matmul(mat1, mat2))
#sums matrices



def matrixAdd(mat1, mat2):
    sumMat = np.zeros((mat1.shape[1],mat2.shape[0]))
    if mat1.shape[1] == mat2.shape[1] and  mat1.shape[0] == mat2.shape[0]:
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                sum = mat1[i][j] +  mat2[i][j]
                
                sumMat[i][j]  = sum

        return sumMat

#subtaracts matrix
def matrixSub(mat1, mat2):
    sumMat = np.zeros((mat1.shape[1],mat2.shape[0]))
    if mat1.shape[1] == mat2.shape[1] and  mat1.shape[0] == mat2.shape[0]:
        for i in range(mat1.shape[0]):
            for j in range(mat1.shape[1]):
                diff = mat1[i][j] -  mat2[i][j]
                
                sumMat[i][j]  = diff 

        return sumMat


#sumM = matrixAdd(mat1, mat2)
@time_dec
def getAdjoint(mat):
    mat = np.array(mat)
    adjMat = np.zeros((mat.shape[1], mat.shape[0]))
    for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                adjMat[j][i] = mat[i][j]
    return adjMat

@time_dec
def getCofactorSubmatrix(mat, row, col):
    mat = np.array(mat)
    subMat = []
    rowlist = []
    minor = getMinor(row, col, mat)
   
    return minor


          

#print(sumM)
#print(np.add(mat1, mat2))


invM = getAdjoint(mat1)
# returns matrix with row removed
def chopRow(row, mat):
    mata = mat[:row , :]
    matb = mat[row+1: , :]
    minor  = np.concatenate((mata, matb), axis = 0)
    return minor

# returns matrix with column removed
def chopCol(col, mat):
    mata = mat[: , :col]
    matb = mat[: ,col+1:]

    minor  = np.concatenate((mata, matb), axis = 1)
    return minor

print(mat1)

tmat = chopRow(3, mat1)
print(tmat)
minor = chopCol(1, tmat)

print(minor)

# returns minor matrix
@time_dec
def getMinor(row, col, mat):
    tmat = chopRow(row, mat)
        
    minor = chopCol(col, tmat)
    return minor


class subMatrix:
    def __init__(self, coef, mat):
        self.coef = coef
        self.mat = mat

    def getMinorList(self):
        subMatList = []
        size = self.mat.shape[0]
        mat = self.mat
        if size > 2:
            for x in range(size):
                Ncoef = mat[0][x]
                
                if x%2 ==0:
                    sign = 1
                else:
                    sign = -1

                NewMat = getMinor(0, x, mat)
                     
                newSub = subMatrix(sign *self.coef*Ncoef, NewMat)
                subMatList.append(newSub)
            
            return subMatList
        else:
            subList = []
            #print(self.mat)
            subList.append(self)
            return subList
@time_dec
def getCofactor(mat):
    mat = np.array(mat)
    cofMat = np.zeros((mat.shape[1],mat.shape[0]))
    #if mat.shape[0] > 2:
   
    for row  in range(mat.shape[0]):
                
            for col in range(mat.shape[0]):
                        submat = getCofactorSubmatrix(mat, row, col)
                        
                        subMat = subMatrix(1, np.array(submat))
                        
                        if subMat.mat.shape[0] > 1:

                            total  = getDeterminate(subMat)
                        else:
                            total = subMat.mat[0]
                        
                        total = total * (-1)**(row+col+2)
                        
                        cofMat[row][col] = total
            
    return cofMat
     
 

testSub = subMatrix(1, mat1)
newList = testSub.getMinorList()

@time_dec
def getDeterminate(subMat):
    #print("determinate twos for two")
    #print(subMat.mat)
    matList = subMat.getMinorList()
  
    twoList = getListOfTwos(matList)
    deter = calcDeterminantFromTwos(twoList)
    return deter
    

@time_dec
def getListOfTwos(matList):
    
    if matList[0].mat.shape[0] >2:
        while(matList[0].mat.shape[0]>2):
            newList = []
            for mat in matList:
                #print("here's matt")
                #print(mat)
                mats = mat.getMinorList()
                #print("two mats")
                #print(mats)
                for item in mats:

                    newList.append(item)
                #print(newList)
            matList = newList
        return matList
    else:
        #print(matList[0].mat)
        #print("list of twos for 2")
        #print(matList[0].mat)
        return matList

twoList = getListOfTwos(newList)



i = 0
#for mat in twoList:
 
    #i += 1

@time_dec
def calcDeterminantFromTwos(matList):
    total= 0
    for mat in matList:
       
        total += mat.coef*(mat.mat[0][0] * mat.mat[1][1] - mat.mat[0][1] * mat.mat[1][0])

    return total

determin =  calcDeterminantFromTwos(twoList)

#print(determin)


cofMat = getCofactor(mat1)
matAdj = np.matrix(mat1)


#print(matAdj.getH())


adjMat = getAdjoint(cofMat)
npCofactor = np.linalg.inv(mat1).T * np.linalg.det(mat1)
#print(mat1)
'''print(npCofactor)

print(cofMat)
#print(adjMat)'''

print(adjMat/determin)

print(np.linalg.inv(mat1))

#adjM = getAdjoint(cofMat)
#print(getDeterminate(testSub))

#print(np.linalg.det(mat1))
