import cython
cimport cython
import numpy
cimport numpy

cdef extern void getJ_c(long int dim, long int batch, double* f, double* F, double* coulGsmall)

@cython.boundscheck(False)
@cython.wraparound(False)

def getJ(long int dim, long int batch, double[:,:] f, double[:,:] F, double[:] coulGsmall):

    getJ_c(dim, batch, &f[0,0], &F[0,0], &coulGsmall[0])

    return None