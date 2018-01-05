import cython
cimport cython
import numpy
cimport numpy

cdef extern void getJ_c(int dim, long int ngs, double* ffunc, double* f2func, double* coulGsmall)

@cython.boundscheck(False)
@cython.wraparound(False)

def getJ(int dim, long int ngs, double[:,:] ffunc, double[:,:] f2func, double[:] coulGsmall):

    getJ_c(dim, ngs, &ffunc[0,0], &f2func[0,0], &coulGsmall[0])

    return None
