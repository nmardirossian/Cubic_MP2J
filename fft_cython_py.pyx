import cython
cimport cython
import numpy
cimport numpy

cdef extern void getJ_c(long int batch, double* F, double* coulGsmall, long int* mesh, long int* smallmesh)
cdef extern double sumtrans_c(long int dim1, long int dim2, double* F1, double* F2, long int* mesh)
cdef extern void mult_c(long int dim1, long int dim2, double* F1, double* F2)

@cython.boundscheck(False)
@cython.wraparound(False)

def getJ(long int batch, double[:,:] F, double[:] coulGsmall, long int[:] mesh, long int[:] smallmesh):

    getJ_c(batch, &F[0,0], &coulGsmall[0], &mesh[0], &smallmesh[0])

    return None

def sumtrans(long int dim1, long int dim2, double[:,:] F1, double[:,:] F2, long int[:] mesh):

    return sumtrans_c(dim1, dim2, &F1[0,0], &F2[0,0], &mesh[0])

def mult(long int dim1, long int dim2, double[:,:] F1, double[:,:] F2):

    mult_c(dim1, dim2, &F1[0,0], &F2[0,0])

    return None
