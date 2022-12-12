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
    mat1 = makeMatrix(3, 4)
    mat2 = makeMatrix(1, 3)
    lin  = MatrixMult(mat1, mat2)
    npmult = np.matmul(mat1, mat2)
    assert(lin.all() == npmult.all())