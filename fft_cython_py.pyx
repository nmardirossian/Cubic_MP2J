import cython
cimport cython
import numpy
cimport numpy

cdef extern void getJ_c(long int batch, double* F, double* coulGsmall, long int* mesh, long int* smallmesh)

@cython.boundscheck(False)
@cython.wraparound(False)

def getJ(long int batch, double[:,:] F, double[:] coulGsmall, long int[:] mesh, long int[:] smallmesh):

    getJ_c(batch, &F[0,0], &coulGsmall[0], &mesh[0], &smallmesh[0])

    return None
