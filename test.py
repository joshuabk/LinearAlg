import numpy as np
from LinearAlg import *


mat1 = makeMatrix(4, 4)

mat2 = makeMatrix(4, 4)

mat3= makeMatrix(6, 6)
def testMult4():
    mat1 = makeMatrix(4, 4)
    mat2 = makeMatrix(4, 4)
    lin  = MatrixMult(mat1, mat2)
    npmult = np.matmul(mat1, mat2)
    assert(lin.all() == npmult.all())


def testMult10():
    mat1 = makeMatrix(10, 10)
    mat2 = makeMatrix(10, 10)
    lin  = MatrixMult(mat1, mat2)
    npmult = np.matmul(mat1, mat2)
    assert(lin.all() == npmult.all())

def testMult105():
    mat1 = makeMatrix(2, 1)
    mat2 = makeMatrix(1, 2)
    lin  = MatrixMult(mat1, mat2)
    npmult = np.matmul(mat1, mat2)
    print(npmult)
    assert(lin.all() == npmult.all())

def testMult34():
    mat1 = makeMatrix(3, 5)
    mat2 = makeMatrix(2, 3)
    lin  = MatrixMult(mat2,mat1 )
    npmult = np.matmul(mat2, mat1)
    assert(lin.all() == npmult.all())

def testMult79():
    mat1 = makeMatrix(7, 9)
    mat2 = makeMatrix(6, 7)
    lin  = MatrixMult(mat2,mat1 )
    npmult = np.matmul(mat2, mat1)
    

def testAdd73():
    mat1 = makeMatrix(7, 3)
    mat2 = makeMatrix(7, 3)
    lin  = matrixAdd(mat1,mat2 )
    npadd = np.add(mat1, mat2)
    assert(lin.all() == npadd.all())

def testAdd54():
    mat1 = makeMatrix(5, 4)
    mat2 = makeMatrix(5, 4)
    lin  = matrixAdd(mat1,mat2 )
    npadd = np.add(mat1, mat2)
    assert(lin.all() == npadd.all())


def testSub38():
    mat1 = makeMatrix(3, 8)
    mat2 = makeMatrix(3, 8)
    lin  = matrixSub(mat2,mat1 )
    npsub = np.subtract(mat2, mat1)
    assert(lin.all() == npsub.all())



def testSub54():
    mat1 = makeMatrix(5, 4)
    mat2 = makeMatrix(5, 4)
    lin  = matrixSub(mat2,mat1 )
    npsub = np.subtract(mat2, mat1)
    assert(lin.all() == npsub.all())