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


mat1 = makeMatrix(2, 2)

mat2 = makeMatrix(4, 4)

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

def getAdjoint(mat):
    mat = np.array(mat)
    adjMat = np.zeros((mat.shape[1], mat.shape[0]))
    for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                adjMat[j][i] = mat[i][j]
    return adjMat

def getCofactorSubmatrix(mat, row, col):
    mat = np.array(mat)
    subMat = []
    rowlist = []
    for i in range(mat.shape[0]):
            for j in range(mat.shape[0]):
                if not (i == row or j == col):
                    rowlist.append(mat[i][j])
            
            if rowlist  != []:
                subMat.append(rowlist)
            rowlist = []
   
    return subMat


    return cofMat        

#print(sumM)
#print(np.add(mat1, mat2))


invM = getAdjoint(mat1)



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

                newSub = subMatrix(sign *self.coef*Ncoef, np.array(NewMat))
                subMatList.append(newSub)
            
            return subMatList
        else:
            subList = []
            subList.append(self)
            return subList

def getCofactor(mat):
    mat = np.array(mat)
    cofMat = np.zeros((mat.shape[1],mat.shape[0]))
    if mat.shape[0] > 2:
        for row  in range(mat.shape[0]):
                
                for col in range(mat.shape[1]):
                    
                    
                        submat = getCofactorSubmatrix(mat, row, col)
                    
                        subMat = subMatrix(1, np.array(submat))
                        total  = getDeterminate(subMat)
                    
                        total = total * (-1)**(row+col+2)
                    
                        cofMat[row][col] = total
            
    return cofMat
     
 

testSub = subMatrix(1, mat1)
newList = testSub.determinate()

def getDeterminate(subMat):
    matList = subMat.determinate()
    
    twoList = getListOfTwos(matList)
    deter = calcDeterminantFromTwos(twoList)
    return deter
    


def getListOfTwos(matList):
    
    if matList[0].mat.shape[0] >2:
        while(matList[0].mat.shape[0]>2):
            newList = []
            for mat in matList:
                for item in mat.determinate():
                    newList.append(item)
            matDetList = newList
        return matList
    else:
        print(matList[0].mat)
        return matList

twoList = getListOfTwos(newList)



i = 0
for mat in twoList:
 
    i += 1

def calcDeterminantFromTwos(matList):
    total= 0
    for mat in matList:
        total += mat.coef*(mat.mat[0][0] * mat.mat[1][1] - mat.mat[0][1] * mat.mat[1][0])

    return total

determin =  calcDeterminantFromTwos(twoList)

print(determin)


cofMat = getCofactor(mat1)
matAdj = np.matrix(mat1)


print(matAdj.getH())


adjMat = getAdjoint(cofMat)
npCofactor = np.linalg.inv(mat1).T * np.linalg.det(mat1)
#print(mat1)
print(npCofactor)
print(cofMat)
#print(adjMat)

#print(adjMat/determin)

#print(np.linalg.inv(mat1))



#adjM = getAdjoint(cofMat)
print(getDeterminate(testSub))

print(np.linalg.det(mat1))
