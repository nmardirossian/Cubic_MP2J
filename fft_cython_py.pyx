import cython
cimport cython
import numpy
cimport numpy

cdef extern void getJ_c(long int batch, complex* F, double* coulG, long int* mesh)

@cython.boundscheck(False)
@cython.wraparound(False)

def getJ(long int batch, complex[:,:] F, double[:] coulG, long int[:] mesh):

    getJ_c(batch, &F[0,0], &coulG[0], &mesh[0])

    return None
