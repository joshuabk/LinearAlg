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
    if not lin.all() == npmult.all():
        print("calc")
        print(lin)
        print("np")
        print(npmult)
    assert(lin.all() == npmult.all())


def testMult10():
    mat1 = makeMatrix(5, 5)
    mat2 = makeMatrix(5, 5)
    lin  = MatrixMult(mat1, mat2)
    npmult = np.matmul(mat1, mat2)
    if not lin.all() == npmult.all():
        print("calc")
        print(lin)
        print("np")
        print(npmult)
    assert(lin.all() == npmult.all())

def testMult105():
    mat1 = makeMatrix(4, 3)
    mat2 = makeMatrix(3, 4)
    lin  = MatrixMult(mat1, mat2)
    npmult = np.matmul(mat1, mat2)
    
    if not lin.all() == npmult.all():
        print("calc")
        print(lin)
        print("np")
        print(npmult)
    assert(lin.all() == npmult.all())

def testMult34():
    mat1 = makeMatrix(3, 5)
    mat2 = makeMatrix(5, 3)
    lin  = MatrixMult(mat2,mat1 )
    npmult = np.matmul(mat2, mat1)
    if not lin.all() == npmult.all():
        print("calc")
        print(lin)
        print("np")
        print(npmult)
    assert(lin.all() == npmult.all())

def testMult79():
    mat1 = makeMatrix(7, 9)
    mat2 = makeMatrix(6, 7)
    lin  = MatrixMult(mat2,mat1 )
    npmult = np.matmul(mat2, mat1)
    if not lin.all() == npmult.all():
        print("calc")
        print(lin)
        print("np")
        print(npmult)
    assert(lin.all() == npmult.all())
    

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


def testDeterminant():
    mat1 = makeMatrix(5, 5)
    subm = subMatrix(1.0, mat1)
    det1 = getDeterminate(subm)
    npdet = np.linalg.det(mat1)
    print(np.linalg.det(mat1))

    assert(round(det1) == round(npdet))

def testCofactor():
    mat1 = makeMatrix(5, 5)
    subm = subMatrix(1.0, mat1)
    print(mat1.shape)
    cofMat = getCofactor(mat1)
    npCofactor = np.linalg.inv(mat1).T * np.linalg.det(mat1)
    assert(cofMat.all() == npCofactor.all())

def testTranspose():
    mat1 = makeMatrix(5, 5)
    trans1 = getTranspose(mat1)
    nptran = mat1.transpose()
    assert(trans1.all() == nptran.all())


def testInverse():
    mat1 = makeMatrix(5, 5)
    subm = subMatrix(1.0, mat1)
    inv1 = getInverse(subm)
    npinv =  np.linalg.inv(mat1)
    assert(inv1.all() == npinv.all())
