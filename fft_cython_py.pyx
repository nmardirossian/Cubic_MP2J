import cython
cimport cython
import numpy
cimport numpy

cdef extern void getJ_c(long int dim, long int batch, double* ffunc, double* f2func, double* coulGsmall)

@cython.boundscheck(False)
@cython.wraparound(False)

def getJ(long int dim, long int batch, double[:,:] ffunc, double[:,:] f2func, double[:] coulGsmall):

    getJ_c(dim, batch, &ffunc[0,0], &f2func[0,0], &coulGsmall[0])

    return None
