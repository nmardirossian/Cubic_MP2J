import cython
cimport cython
import numpy
cimport numpy

cdef extern void fftreal_c(long int batch, double* F, double* coulG, long int* mesh, long int* smallmesh)
cdef extern void fftcomp_c(long int batch, complex* F, double* coulG, long int* mesh)
cdef extern double sum_c(long int dim1, long int dim2, long int p1, long int p2, double* F1, double* F2)
cdef extern double sumtrans_c(long int dim1, long int dim2, long int p1, long int p2, double* F1, double* F2)
cdef extern void mult_c(long int dim1, long int dim2, double* F1, double* F2)
cdef extern void trans_c(long int dim1, long int dim2, long int ngs, double* F1, double* F2)

@cython.boundscheck(False)
@cython.wraparound(False)

def fftreal(long int batch, double[:,:] F, double[:] coulG, long int[:] mesh, long int[:] smallmesh):

    fftreal_c(batch, &F[0,0], &coulG[0], &mesh[0], &smallmesh[0])

    return None

def fftcomp(long int batch, complex[:,:] F, double[:] coulG, long int[:] mesh):

    fftcomp_c(batch, &F[0,0], &coulG[0], &mesh[0])

    return None

def sum(long int dim1, long int dim2, long int p1, long int p2, double[:,:] F1, double[:,:] F2):

    return sum_c(dim1, dim2, p1, p2, &F1[0,0], &F2[0,0])

def sumtrans(long int dim1, long int dim2, long int p1, long int p2, double[:,:] F1, double[:,:] F2):

    return sumtrans_c(dim1, dim2, p1, p2, &F1[0,0], &F2[0,0])

def mult(long int dim1, long int dim2, double[:,:] F1, double[:,:] F2):

    mult_c(dim1, dim2, &F1[0,0], &F2[0,0])

    return None

def trans(long int dim1, long int dim2, long int ngs, double[:,:] F1, double[:,:] F2):

    trans_c(dim1, dim2, ngs, &F1[0,0], &F2[0,0])

    return None
